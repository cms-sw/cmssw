#include "DQMOffline/Hcal/interface/CaloTowersDQMClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

CaloTowersDQMClient::CaloTowersDQMClient(const edm::ParameterSet &iConfig) : conf_(iConfig) {
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  debug_ = false;
  verbose_ = false;
  dirName_ = iConfig.getParameter<std::string>("DQMDirName");
}

CaloTowersDQMClient::~CaloTowersDQMClient() {}

void CaloTowersDQMClient::beginJob() {}

void CaloTowersDQMClient::beginRun(const edm::Run &run, const edm::EventSetup &c) {}

// called after entering the CaloTowersD/CaloTowersTask directory
// hcalMEs are within that directory
int CaloTowersDQMClient::CaloTowersEndjob(const std::vector<MonitorElement *> &hcalMEs) {
  int useAllHistos = 0;
  MonitorElement *Ntowers_vs_ieta = nullptr;
  MonitorElement *mapEnergy_N = nullptr, *mapEnergy_E = nullptr, *mapEnergy_H = nullptr, *mapEnergy_EH = nullptr;
  MonitorElement *occupancy_map = nullptr, *occupancy_vs_ieta = nullptr;
  for (unsigned int ih = 0; ih < hcalMEs.size(); ih++) {
    if (strcmp(hcalMEs[ih]->getName().c_str(), "Ntowers_per_event_vs_ieta") == 0) {
      Ntowers_vs_ieta = hcalMEs[ih];
    }
    if (strcmp(hcalMEs[ih]->getName().c_str(), "CaloTowersTask_map_Nentries") == 0) {
      mapEnergy_N = hcalMEs[ih];
    }
    if (strcmp(hcalMEs[ih]->getName().c_str(), "CaloTowersTask_map_energy_H") == 0) {
      useAllHistos++;
      mapEnergy_H = hcalMEs[ih];
    }
    if (strcmp(hcalMEs[ih]->getName().c_str(), "CaloTowersTask_map_energy_E") == 0) {
      useAllHistos++;
      mapEnergy_E = hcalMEs[ih];
    }
    if (strcmp(hcalMEs[ih]->getName().c_str(), "CaloTowersTask_map_energy_EH") == 0) {
      useAllHistos++;
      mapEnergy_EH = hcalMEs[ih];
    }
    if (strcmp(hcalMEs[ih]->getName().c_str(), "CaloTowersTask_map_occupancy") == 0) {
      occupancy_map = hcalMEs[ih];
    }
    if (strcmp(hcalMEs[ih]->getName().c_str(), "CaloTowersTask_occupancy_vs_ieta") == 0) {
      occupancy_vs_ieta = hcalMEs[ih];
    }
  }
  if (useAllHistos != 0 && useAllHistos != 3)
    return 0;

  double nevent = mapEnergy_N->getEntries();
  if (verbose_)
    std::cout << "nevent : " << nevent << std::endl;

  // mean number of towers per ieta
  int nx = Ntowers_vs_ieta->getNbinsX();
  float cont;
  float conte;
  float fev = float(nevent);

  for (int i = 1; i <= nx; i++) {
    cont = Ntowers_vs_ieta->getBinContent(i) / fev;
    conte = pow(Ntowers_vs_ieta->getBinContent(i), 0.5) / fev;
    Ntowers_vs_ieta->setBinContent(i, cont);
    Ntowers_vs_ieta->setBinError(i, conte);
  }

  // mean energies & occupancies evaluation

  nx = mapEnergy_N->getNbinsX();
  int ny = mapEnergy_N->getNbinsY();
  float cnorm;
  float cnorme;
  float phi_factor;

  for (int i = 1; i <= nx; i++) {
    float sumphi = 0.;

    for (int j = 1; j <= ny; j++) {
      // Emean
      cnorm = mapEnergy_N->getBinContent(i, j);
      // Phi histos are not used in the macros
      if (cnorm > 0.000001 && useAllHistos) {
        cont = mapEnergy_E->getBinContent(i, j) / cnorm;
        conte = mapEnergy_E->getBinError(i, j) / cnorm;
        mapEnergy_E->setBinContent(i, j, cont);
        mapEnergy_E->setBinError(i, j, conte);

        cont = mapEnergy_H->getBinContent(i, j) / cnorm;
        conte = mapEnergy_H->getBinError(i, j) / cnorm;
        mapEnergy_H->setBinContent(i, j, cont);
        mapEnergy_H->setBinError(i, j, conte);

        cont = mapEnergy_EH->getBinContent(i, j) / cnorm;
        conte = mapEnergy_EH->getBinError(i, j) / cnorm;
        mapEnergy_EH->setBinContent(i, j, cont);
        mapEnergy_EH->setBinError(i, j, conte);
      }

      // Occupancy (needed for occupancy vs ieta)
      cont = occupancy_map->getBinContent(i, j);
      conte = occupancy_map->getBinError(i, j);
      if (fev > 0. && cnorm > 1.e-30) {
        occupancy_map->setBinContent(i, j, cont / fev);
        occupancy_map->setBinError(i, j, conte / fev);
      }

      sumphi += cont;

    }  // end of iphy cycle (j)

    // Occupancy vs ieta histo is drawn
    // phi-factor evaluation for occupancy_vs_ieta calculation
    int ieta = i - 43;  // should be the same as int ieta =
                        // int(occupancy_vs_ieta->getBinCenter(i));

    if (ieta >= -20 && ieta <= 20) {
      phi_factor = 72.;
    } else {
      if (ieta >= 40 || ieta <= -40) {
        phi_factor = 18.;
      } else
        phi_factor = 36.;
    }

    cnorm = sumphi / phi_factor;
    cnorme = pow(sumphi, 0.5) / phi_factor;
    if (fev > 0. && cnorm > 1.e-30) {
      occupancy_vs_ieta->setBinContent(i, cnorm / fev);
      occupancy_vs_ieta->setBinError(i, cnorme / fev);
    }

  }  // end of ieta cycle (i)

  return 1;
}

void CaloTowersDQMClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  igetter.setCurrentFolder(dirName_);
  if (verbose_)
    std::cout << "\nrunClient" << std::endl;

  std::vector<MonitorElement *> hcalMEs;

  // Since out folders are fixed to three, we can just go over these three
  // folders i.e., CaloTowersD/CaloTowersTask, HcalRecHitsD/HcalRecHitTask,
  // NoiseRatesV/NoiseRatesTask.
  std::vector<std::string> fullPathHLTFolders = igetter.getSubdirs();
  for (unsigned int i = 0; i < fullPathHLTFolders.size(); i++) {
    if (verbose_)
      std::cout << "\nfullPath: " << fullPathHLTFolders[i] << std::endl;
    igetter.setCurrentFolder(fullPathHLTFolders[i]);

    std::vector<std::string> fullSubPathHLTFolders = igetter.getSubdirs();
    for (unsigned int j = 0; j < fullSubPathHLTFolders.size(); j++) {
      if (verbose_)
        std::cout << "fullSub: " << fullSubPathHLTFolders[j] << std::endl;

      if (strcmp(fullSubPathHLTFolders[j].c_str(), "CaloTowersD/CaloTowersTask") == 0) {
        hcalMEs = igetter.getContents(fullSubPathHLTFolders[j]);
        if (verbose_)
          std::cout << "hltMES size : " << hcalMEs.size() << std::endl;
        if (!CaloTowersEndjob(hcalMEs))
          std::cout << "\nError in CaloTowersEndjob!" << std::endl << std::endl;
      }
    }
  }
}

DEFINE_FWK_MODULE(CaloTowersDQMClient);
