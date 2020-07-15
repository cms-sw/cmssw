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

#include "CondFormats/CSCObjects/interface/CSCBadStrips.h"
#include "CondFormats/DataRecord/interface/CSCBadStripsRcd.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

namespace edmtest {
  class CSCReadBadStripsAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCReadBadStripsAnalyzer(edm::ParameterSet const& ps)
        : outputToFile(ps.getParameter<bool>("outputToFile")),
          readBadChannels_(ps.getParameter<bool>("readBadChannels")) {
      badStripWords.resize(3240, 0);
    }

    explicit CSCReadBadStripsAnalyzer(int i) {}
    ~CSCReadBadStripsAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

    // Test code from CSCConditions

    /// did we request reading bad channel info from db?
    bool readBadChannels() const { return readBadChannels_; }

    void fillBadStripWords();

    /// return  bad channel words per CSCLayer - 1 bit per channel
    const std::bitset<80>& badStripWord(const CSCDetId& id) const;

  private:
    bool outputToFile;
    bool readBadChannels_;  // flag whether or not to even attempt reading bad channel info from db
    const CSCBadStrips* theBadStrips;

    std::vector<std::bitset<80> > badStripWords;
  };

  void CSCReadBadStripsAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;

    int counter = 0;
    std::cout << " RUN# " << e.id().run() << std::endl;
    std::cout << " EVENT# " << e.id().event() << std::endl;
    edm::ESHandle<CSCBadStrips> pBadStrip;
    context.get<CSCBadStripsRcd>().get(pBadStrip);

    theBadStrips = pBadStrip.product();

    // Create the vectors of bitsets, one per layer, as done in CSCConditions
    fillBadStripWords();  // code from CSCConditions pasted into this file!

    CSCIndexer indexer;  // just to build a CSCDetId from chamber index

    std::vector<CSCBadStrips::BadChamber>::const_iterator itcham;
    std::vector<CSCBadStrips::BadChannel>::const_iterator itchan;

    std::cout << "Bad Chambers:" << std::endl;

    int ibad = 0;
    int ifailed = 0;

    // Iterate over the list of bad chambers

    for (itcham = theBadStrips->chambers.begin(); itcham != theBadStrips->chambers.end(); ++itcham) {
      counter++;
      int indexc = itcham->chamber_index;
      int badstart = itcham->pointer;
      int nbad = itcham->bad_channels;
      std::cout << counter << "  " << itcham->chamber_index << "  " << itcham->pointer << "  " << itcham->bad_channels
                << std::endl;
      CSCDetId id = indexer.detIdFromChamberIndex(indexc);

      // Iterate over the bad channels in this chamber

      for (int ichan = badstart - 1; ichan != badstart - 1 + nbad; ++ichan) {
        short lay = theBadStrips->channels[ichan].layer;
        short chan = theBadStrips->channels[ichan].channel;

        // create a CSCDetId for this layer, just because that it is needed for the interface to CSCConditions::badStripWord
        CSCDetId id2 = CSCDetId(id.endcap(), id.station(), id.ring(), id.chamber(), lay);
        std::bitset<80> ibits = badStripWord(id2);

        // Test whether this bad channel has indeed been flagged in the badStripWord
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

    for( itchan=theBadStrips->channels.begin();itchan!=theBadStrips->channels.end(); ++itchan ){    
      counter++;
      std::cout<<counter<<"  "<<itchan->layer<<"  "<<itchan->channel<<"  "<<itchan->flag1<<std::endl;
    }
    */

    if (outputToFile) {
      std::ofstream BadStripFile("dbBadStrip.dat", std::ios::app);

      counter = 0;
      for (itcham = theBadStrips->chambers.begin(); itcham != theBadStrips->chambers.end(); ++itcham) {
        counter++;
        BadStripFile << counter << "  " << itcham->chamber_index << "  " << itcham->pointer << "  "
                     << itcham->bad_channels << std::endl;
      }
      counter = 0;
      for (itchan = theBadStrips->channels.begin(); itchan != theBadStrips->channels.end(); ++itchan) {
        counter++;
        BadStripFile << counter << "  " << itchan->layer << "  " << itchan->channel << "  " << itchan->flag1
                     << std::endl;
      }
    }
  }

  void CSCReadBadStripsAnalyzer::fillBadStripWords() {
    // reset existing values
    badStripWords.assign(3240, 0);
    if (readBadChannels()) {
      // unpack what we read from theBadStrips

      // chambers is a vector<BadChamber>
      // channels is a vector<BadChannel>
      // Each BadChamber contains its index (1-468 or 540 w. ME42), the no. of bad channels,
      // and the index within vector<BadChannel> where this chamber's bad channels start.

      CSCIndexer indexer;

      int icount = 0;

      for (size_t i = 0; i < theBadStrips->chambers.size(); ++i) {  // loop over bad chambers
        int indexc = theBadStrips->chambers[i].chamber_index;

        // The following is not in standard CSCConditions version but was required for our prototype bad strip db
        if (indexc == 0) {
          std::cout << "WARNING: chamber index = 0. Quitting. " << std::endl;
          break;  // prototype db has zero line at end
        }

        int start = theBadStrips->chambers[i].pointer;  // where this chamber's bad channels start in vector<BadChannel>
        int nbad = theBadStrips->chambers[i].bad_channels;

        CSCDetId id = indexer.detIdFromChamberIndex(indexc);  // We need this to build layer index (1-3240)

        for (int j = start - 1; j < start + nbad - 1; ++j) {  // bad channels in this chamber
          short lay = theBadStrips->channels[j].layer;        // value 1-6

          // The following is not in standard CSCConditions version but was required for our prototype bad strip db
          if (lay == 0) {
            std::cout << "WARNING: layer index = 0. Quitting. " << std::endl;
            break;
          }

          short chan = theBadStrips->channels[j].channel;  // value 1-80
                                                           //    short f1 = theBadStrips->channels[j].flag1;
                                                           //    short f2 = theBadStrips->channels[j].flag2;
                                                           //    short f3 = theBadStrips->channels[j].flag3;
          int indexl = indexer.layerIndex(id.endcap(), id.station(), id.ring(), id.chamber(), lay);

          // Test output to monitor filling
          std::cout << "count " << ++icount << " bad channel " << chan << " in layer " << lay << " of chamber=" << id
                    << " chamber index=" << indexc << " layer index=" << indexl << std::endl;

          badStripWords[indexl - 1].set(chan - 1, true);  // set bit in 80-bit bitset representing this layer
        }                                                 // j
      }                                                   // i
    }
  }

  const std::bitset<80>& CSCReadBadStripsAnalyzer::badStripWord(const CSCDetId& id) const {
    CSCIndexer indexer;
    return badStripWords[indexer.layerIndex(id) - 1];
  }

  DEFINE_FWK_MODULE(CSCReadBadStripsAnalyzer);
}  // namespace edmtest
