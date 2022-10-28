/**
 * \file EcalGraphDumperModule.h 
 * module dumping TGraph with 10 data frames
 *   
 * 
 * \author N. Amapane - S. Argiro'
 * \author G. Franzoni
 *
 */

#include <FWCore/Framework/interface/one/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>

#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>

#include <iostream>
#include <vector>

#include "TFile.h"
#include "TGraph.h"

class EcalGraphDumperModule : public edm::one::EDAnalyzer<> {
public:
  EcalGraphDumperModule(const edm::ParameterSet& ps);
  ~EcalGraphDumperModule();

  std::string intToString(int num);

  void analyze(const edm::Event& e, const edm::EventSetup& c);

protected:
  int verbosity;
  int eventCounter;

  int ieb_id;
  int first_ic;

  bool inputIsOk;

  std::string fileName;

  std::vector<int> listChannels;
  std::vector<int> listAllChannels;
  std::vector<int> listPns;

  int side;

  int abscissa[10];
  int ordinate[10];

  std::vector<TGraph> graphs;

  TFile* root_file;
};

EcalGraphDumperModule::EcalGraphDumperModule(const edm::ParameterSet& ps) {
  fileName = ps.getUntrackedParameter<std::string>("fileName", std::string("toto"));

  ieb_id = ps.getUntrackedParameter<int>("ieb_id", 1);
  first_ic = 0;

  listChannels = ps.getUntrackedParameter<std::vector<int> >("listChannels", std::vector<int>());

  side = ps.getUntrackedParameter<int>("side", 3);

  // consistency checks checks
  inputIsOk = true;

  std::vector<int>::iterator intIter;

  for (intIter = listChannels.begin(); intIter != listChannels.end(); intIter++) {
    if (((*intIter) < 1) || (1700 < (*intIter))) {
      std::cout << "[EcalGraphDumperModule] ic value: " << (*intIter) << " found in listChannels. "
                << " Valid range is 1-1700. Returning." << std::endl;
      inputIsOk = false;
      return;
    }
    // initializing with the first channel of the list
    if (!first_ic)
      first_ic = (*intIter);
  }

  // setting the abcissa array once for all
  for (int i = 0; i < 10; i++)
    abscissa[i] = i;

  // local event counter (in general different from LV1)
  eventCounter = 0;
}

EcalGraphDumperModule::~EcalGraphDumperModule() {
  fileName += (std::string("_iEB") + intToString(ieb_id));
  fileName += (std::string("_ic") + intToString(first_ic));
  fileName += ".graph.root";

  root_file = new TFile(fileName.c_str(), "RECREATE");
  std::vector<TGraph>::iterator gr_it;
  for (gr_it = graphs.begin(); gr_it != graphs.end(); gr_it++)
    (*gr_it).Write();
  root_file->Close();
}

std::string EcalGraphDumperModule::intToString(int num) {
  //
  // outputs the number into the string stream and then flushes
  // the buffer (makes sure the output is put into the stream)
  //
  std::ostringstream myStream;  //creates an ostringstream object
  myStream << num << std::flush;

  return (myStream.str());  //returns the string form of the stringstream object
}

void EcalGraphDumperModule::analyze(const edm::Event& e, const edm::EventSetup& c) {
  eventCounter++;
  if (!inputIsOk)
    return;

  // retrieving crystal data from Event
  edm::Handle<EBDigiCollection> digis;
  e.getByLabel("ecalEBunpacker", "ebDigis", digis);

  // retrieving crystal PN diodes from Event
  edm::Handle<EcalPnDiodeDigiCollection> PNs;
  e.getByLabel("ecalEBunpacker", PNs);

  // getting the list of all the channels which will be dumped on TGraph
  std::vector<int>::iterator ch_it;
  for (ch_it = listChannels.begin(); ch_it != listChannels.end(); ch_it++) {
    int ic = (*ch_it);
    int ieta = (ic - 1) / 20 + 1;
    int iphi = (ic - 1) % 20 + 1;

    int hside = (side / 2);

    for (int u = (-hside); u <= hside; u++) {
      for (int v = (-hside); v <= hside; v++) {
        int ieta_c = ieta + u;
        int iphi_c = iphi + v;

        if (ieta_c < 1 || 85 < ieta_c)
          continue;
        if (iphi_c < 1 || 20 < iphi_c)
          continue;

        int ic_c = (ieta_c - 1) * 20 + iphi_c;
        listAllChannels.push_back(ic_c);
      }
    }
  }

  for (EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr) {
    {
      int ic = EBDetId((*digiItr).id()).ic();
      int ieb = EBDetId((*digiItr).id()).ism();
      if (ieb != ieb_id)
        return;

      // selecting desired channels only
      std::vector<int>::iterator icIter;
      icIter = find(listAllChannels.begin(), listAllChannels.end(), ic);
      if (icIter == listAllChannels.end()) {
        continue;
      }

      for (int i = 0; i < ((int)(*digiItr).size()); ++i) {
        EBDataFrame df(*digiItr);
        ordinate[i] = df.sample(i).adc();
      }

      TGraph oneGraph(10, abscissa, ordinate);
      std::string title;
      title = "Graph_ev" + intToString(eventCounter) + "_ic" + intToString(ic);
      oneGraph.SetTitle(title.c_str());
      oneGraph.SetName(title.c_str());
      graphs.push_back(oneGraph);

    }  // loop in crystals
  }
}

DEFINE_FWK_MODULE(EcalGraphDumperModule);
