#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "EventFilter/L1TRawToDigi/interface/L1TRawToDigi.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   L1TRawToDigi::L1TRawToDigi(const edm::ParameterSet& config) :
      inputLabel_(config.getParameter<edm::InputTag>("InputLabel")),
      fedId_(config.getParameter<int>("FedId"))
   {
      auto factory_names = config.getParameter<std::vector<std::string>>("Unpackers");
      for (const auto& name: factory_names)
         factories_.push_back(UnpackerFactory::get()->makeUnpackerFactory(name, config, *this));
   }


   L1TRawToDigi::~L1TRawToDigi()
   {
   }


   //
   // member functions
   //

   // ------------ method called to produce the data  ------------
   void
   L1TRawToDigi::produce(edm::Event& event, const edm::EventSetup& setup)
   {
      using namespace edm;

      edm::Handle<FEDRawDataCollection> feds;
      event.getByLabel(inputLabel_, feds);

      if (!feds.isValid()) {
         LogError("L1T") << "Cannot unpack: no collection found with input tag "
            << inputLabel_;
         return;
      }

      const FEDRawData& l1tRcd = feds->FEDData(fedId_);

      if (l1tRcd.size() < 20) {
         LogError("L1T") << "Cannot unpack: empty/invalid L1T raw data (size = "
            << l1tRcd.size() << "). Returning empty collections!";
         return;
      }

      const unsigned char *data = l1tRcd.data();
      // FIXME is the 8 right?
      // const unsigned data_end = l1tRcd.size() - 8;
      unsigned idx = 16;

      // Extract header data
      // uint32_t event_id = pop(data, idx) & 0xFFFFFF;

      uint32_t id = pop(data, idx);
      id = pop(data, idx);
      // uint32_t bx_id = (id >> 16) & 0xFFF;
      // uint32_t orbit_id = (id >> 4) & 0x1F;
      // uint32_t board_id = id & 0xF;

      id = pop(data, idx);
      uint32_t fw_id = (id >> 24) & 0xFF;
      uint32_t payload_size = (id >> 8) & 0xFFFF;
      // uint32_t event_type = id & 0xFF;

      if (l1tRcd.size() < payload_size * 4 + 36) {
         LogError("L1T") << "Cannot unpack: invalid L1T raw data size in header (size = "
            << l1tRcd.size() << ", expected " << payload_size * 4 + 36
            << " + padding). Returning empty collections!";
         return;
      }

      unsigned fw = fw_id;

      UnpackerMap unpackers;
      for (auto& f: factories_) {
         for (const auto& up: f->create(event, fedId_, fw)) {
            unpackers.insert(up);
         }
      }

      auto payload_end = idx + payload_size * 4;
      for (unsigned int b = 0; idx < payload_end; ++b) {
         // FIXME Number of blocks actually fixed by firmware
         if (b >= MAX_BLOCKS) {
            LogError("L1T") << "Reached block limit - bailing out from this event!";
            break;
         }

         // Parse block
         uint32_t block_hdr = pop(data, idx);
         uint32_t block_id = (block_hdr >> 24) & 0xFF;
         uint32_t block_size = (block_hdr >> 16) & 0xFF;

         auto unpacker = unpackers.find(block_id);
         if (unpacker == unpackers.end()) {
            LogWarning("L1T") << "Cannot find an unpacker for block ID "
               << block_id << ", FED ID " << fedId_ << ", and FW ID "
               << fw << "!";
            // TODO Handle error
         } else if (!unpacker->second->unpack(data + idx, block_id, block_size)) {
            LogWarning("L1T") << "Error unpacking data for block ID "
               << block_id << ", FED ID " << fedId_ << ", and FW ID "
               << fw << "!";
            // TODO Handle error
         }

         // Advance index by block size (header does this in pop())
         idx += block_size * 4;
      }
   }

   // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
   void
   L1TRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
     //The following says we do not know what parameters are allowed so do no validation
     // Please change this to state exactly what you do use, even if it is no parameters
     edm::ParameterSetDescription desc;
     desc.setUnknown();
     descriptions.addDefault(desc);
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1TRawToDigi);
