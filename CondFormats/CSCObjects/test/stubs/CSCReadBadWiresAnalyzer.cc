#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <bitset>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCBadWires.h"
#include "CondFormats/DataRecord/interface/CSCBadWiresRcd.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

namespace edmtest {
  class CSCReadBadWiresAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCReadBadWiresAnalyzer(edm::ParameterSet const& ps)
        : outputToFile(ps.getParameter<bool>("outputToFile")),
          readBadChannels_(ps.getParameter<bool>("readBadChannels")) {
      badWireWords.resize(3240, 0);  // incl. ME42
    }

    explicit CSCReadBadWiresAnalyzer(int i) {}
    ~CSCReadBadWiresAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

    // Test code from CSCConditions

    /// did we request reading bad channel info from db?
    bool readBadChannels() const { return readBadChannels_; }

    void fillBadWireWords();

    /// return  bad channel words per CSCLayer - 1 bit per channel
    const std::bitset<112>& badWireWord(const CSCDetId& id) const;

  private:
    bool outputToFile;
    bool readBadChannels_;  // flag whether or not to even attempt reading bad channel info from db
    const CSCBadWires* theBadWires;

    std::vector<std::bitset<112> > badWireWords;
  };

  void CSCReadBadWiresAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;

    int counter = 0;
    std::cout << " RUN# " << e.id().run() << std::endl;
    std::cout << " EVENT# " << e.id().event() << std::endl;
    edm::ESHandle<CSCBadWires> pBadWire;
    context.get<CSCBadWiresRcd>().get(pBadWire);

    theBadWires = pBadWire.product();

    // Create the vectors of bitsets, one per layer, as done in CSCConditions
    fillBadWireWords();  // code from CSCConditions pasted into this file!

    CSCIndexer indexer;  // just to build a CSCDetId from chamber index

    std::vector<CSCBadWires::BadChamber>::const_iterator itcham;
    std::vector<CSCBadWires::BadChannel>::const_iterator itchan;

    std::cout << "Bad Chambers:" << std::endl;

    int ibad = 0;
    int ifailed = 0;

    // Iterate over the list of bad chambers

    for (itcham = theBadWires->chambers.begin(); itcham != theBadWires->chambers.end(); ++itcham) {
      counter++;
      int indexc = itcham->chamber_index;
      int badstart = itcham->pointer;
      int nbad = itcham->bad_channels;
      std::cout << counter << "  " << itcham->chamber_index << "  " << itcham->pointer << "  " << itcham->bad_channels
                << std::endl;
      CSCDetId id = indexer.detIdFromChamberIndex(indexc);

      // Iterate over the bad channels in this chamber

      for (int ichan = badstart - 1; ichan != badstart - 1 + nbad; ++ichan) {
        short lay = theBadWires->channels[ichan].layer;
        short chan = theBadWires->channels[ichan].channel;

        // create a CSCDetId for this layer, just because that it is needed for the interface to CSCConditions::badWireWord
        CSCDetId id2 = CSCDetId(id.endcap(), id.station(), id.ring(), id.chamber(), lay);
        std::bitset<112> ibits = badWireWord(id2);

        // Test whether this bad channel has indeed been flagged in the badWireWord
        if (ibits.test(chan - 1)) {
          std::cout << "count " << ++ibad << " found bad channel " << chan << " in layer " << id2 << std::endl;
        } else {
          std::cout << "count " << +ifailed << " failed to see bad channel " << chan << " in layer " << id2
                    << std::endl;
        }
      }
    }

    /*
    std::cout<< "Bad Channels:" << std::endl;
    counter = 0; // reset it!

    for( itchan=theBadWires->channels.begin();itchan!=theBadWires->channels.end(); ++itchan ){    
      counter++;
      std::cout<<counter<<"  "<<itchan->layer<<"  "<<itchan->channel<<"  "<<itchan->flag1<<std::endl;
    }
    */

    if (outputToFile) {
      std::ofstream BadWireFile("dbBadWire.dat", std::ios::app);

      counter = 0;
      for (itcham = theBadWires->chambers.begin(); itcham != theBadWires->chambers.end(); ++itcham) {
        counter++;
        BadWireFile << counter << "  " << itcham->chamber_index << "  " << itcham->pointer << "  "
                    << itcham->bad_channels << std::endl;
      }
      counter = 0;
      for (itchan = theBadWires->channels.begin(); itchan != theBadWires->channels.end(); ++itchan) {
        counter++;
        BadWireFile << counter << "  " << itchan->layer << "  " << itchan->channel << "  " << itchan->flag1
                    << std::endl;
      }
    }
  }

  void CSCReadBadWiresAnalyzer::fillBadWireWords() {
    // reset existing values
    badWireWords.assign(3240, 0);
    if (readBadChannels()) {
      // unpack what we read from theBadWires

      // chambers is a vector<BadChamber>
      // channels is a vector<BadChannel>
      // Each BadChamber contains its index (1-468 or 540 w. ME42), the no. of bad channels,
      // and the index within vector<BadChannel> where this chamber's bad channels start.

      CSCIndexer indexer;

      int icount = 0;

      for (size_t i = 0; i < theBadWires->chambers.size(); ++i) {  // loop over bad chambers
        int indexc = theBadWires->chambers[i].chamber_index;

        // The following is not in standard CSCConditions version but was required for our prototype bad strip db
        if (indexc == 0) {
          std::cout << "WARNING: chamber index = 0. Quitting. " << std::endl;
          break;  // prototype db has zero line at end
        }

        int start = theBadWires->chambers[i].pointer;  // where this chamber's bad channels start in vector<BadChannel>
        int nbad = theBadWires->chambers[i].bad_channels;

        CSCDetId id = indexer.detIdFromChamberIndex(indexc);  // We need this to build layer index (1-3240)

        for (int j = start - 1; j < start + nbad - 1; ++j) {  // bad channels in this chamber
          short lay = theBadWires->channels[j].layer;         // value 1-6

          // The following is not in standard CSCConditions version but was required for our prototype bad strip db
          if (lay == 0) {
            std::cout << "WARNING: layer index = 0. Quitting. " << std::endl;
            break;
          }

          short chan = theBadWires->channels[j].channel;  // value 1-112
                                                          //    short f1 = theBadWires->channels[j].flag1;
                                                          //    short f2 = theBadWires->channels[j].flag2;
                                                          //    short f3 = theBadWires->channels[j].flag3;
          int indexl = indexer.layerIndex(id.endcap(), id.station(), id.ring(), id.chamber(), lay);

          // Test output to monitor filling
          std::cout << "count " << ++icount << " bad channel " << chan << " in layer " << lay << " of chamber=" << id
                    << " chamber index=" << indexc << " layer index=" << indexl << std::endl;

          badWireWords[indexl - 1].set(chan - 1, true);  // set bit in 112-bit bitset representing this layer
        }                                             // j
      }                                               // i
    }
  }

  const std::bitset<112>& CSCReadBadWiresAnalyzer::badWireWord(const CSCDetId& id) const {
    CSCIndexer indexer;
    return badWireWords[indexer.layerIndex(id) - 1];
  }

  DEFINE_FWK_MODULE(CSCReadBadWiresAnalyzer);
}  // namespace edmtest
