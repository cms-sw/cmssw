#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "DataFormats/L1Trigger/interface/Jet.h"

#include "EventFilter/L1TRawToDigi/interface/L1TDigiToRaw.h"

#include "FWCore/Utilities/interface/CRC16.h"

#include <iomanip>

namespace l1t {
   L1TDigiToRaw::L1TDigiToRaw(const edm::ParameterSet& config) :
      inputLabel_(config.getParameter<edm::InputTag>("InputLabel")),
      fedId_(config.getParameter<int>("FedId"))
   {
      // Register products
      produces<FEDRawDataCollection>();
   }


   L1TDigiToRaw::~L1TDigiToRaw()
   {
   }


   //
   // member functions
   //

   inline uint32_t pop(const unsigned char* ptr, unsigned& idx) {
      uint32_t res = ptr[idx + 0] | (ptr[idx + 1] << 4) | (ptr[idx + 2] << 8) | (ptr[idx + 3] << 12);
      idx += 4;
      return res;
   }

   inline unsigned char *
   push(unsigned char * ptr, uint32_t word)
   {
      ptr[0] = word & 0xFF;
      ptr[1] = (word >> 8) & 0xFF;
      ptr[2] = (word >> 16) & 0xFF;
      ptr[3] = (word >> 24) & 0xFF;
      return ptr + 4;
   }

   // ------------ method called to produce the data  ------------
   void
   L1TDigiToRaw::produce(edm::Event& event, const edm::EventSetup& setup)
   {
      using namespace edm;

      edm::Handle<l1t::JetBxCollection> jets;
      event.getByLabel(inputLabel_, jets);

      std::auto_ptr<FEDRawDataCollection> raw_coll(new FEDRawDataCollection());
      FEDRawData& fed_data = raw_coll->FEDData(fedId_);

      unsigned int size = 24; // 16 for the header, 8 for the footer
      size += 12; // the L1T header...
      unsigned int njets = 0;
      for (int i = jets->getFirstBX(); i <= jets->getLastBX(); ++i) {
         njets += jets->size(i);
      }
      size += njets * 7 * 4; // 6 words for members, 1 for block header
      int mod = size % 8;
      size = (mod == 0) ? size : size + 8 - mod;
      fed_data.resize(size);
      unsigned char * header = fed_data.data();
      unsigned char * payload = header + 16;
      unsigned char * footer = header + size - 8;

      FEDHeader fed_header(header);
      // FIXME the 0 is the BX id
      fed_header.set(header, 1, event.id().event(), 0, fedId_);

      // create the header
      payload = push(payload, 0);
      payload = push(payload, 0);
      payload = push(payload, (1 << 24) | ((njets * 7) << 8)); // FW ID, payload size (words)

      for (int bx = jets->getFirstBX(); bx <= jets->getLastBX(); ++bx) {
         for (unsigned int i = 0; i < jets->size(bx); ++i) {
            payload = push(payload, (0xF << 24) | (6 << 16)); // block ID, block size (words)
            payload = push(payload, bx);
            payload = push(payload, jets->at(bx, i).hwPt());
            payload = push(payload, jets->at(bx, i).hwEta());
            payload = push(payload, jets->at(bx, i).hwPhi());
            payload = push(payload, jets->at(bx, i).hwQual());
            payload = push(payload, jets->at(bx, i).hwIso());
         }
      }

      FEDTrailer trailer(footer);
      trailer.set(footer, size / 8, evf::compute_crc(header, size), 0, 0);

      event.put(raw_coll);
   }

   // ------------ method called once each job just before starashtting event loop  ------------
   void 
   L1TDigiToRaw::beginJob()
   {
   }

   // ------------ method called once each job just after ending the event loop  ------------
   void 
   L1TDigiToRaw::endJob() {
   }

   // ------------ method called when starting to processes a run  ------------
   /*
   void
   L1TDigiToRaw::beginRun(edm::Run const&, edm::EventSetup const&)
   {
   }
   */
    
   // ------------ method called when ending the processing of a run  ------------
   /*
   void
   L1TDigiToRaw::endRun(edm::Run const&, edm::EventSetup const&)
   {
   }
   */
    
   // ------------ method called when starting to processes a luminosity block  ------------
   /*
   void
   L1TDigiToRaw::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
   {
   }
   */
    
   // ------------ method called when ending the processing of a luminosity block  ------------
   /*
   void
   L1TDigiToRaw::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
   {
   }
   */
    
   // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
   void
   L1TDigiToRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
     //The following says we do not know what parameters are allowed so do no validation
     // Please change this to state exactly what you do use, even if it is no parameters
     edm::ParameterSetDescription desc;
     desc.setUnknown();
     descriptions.addDefault(desc);
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1TDigiToRaw);
