// -*- C++ -*-
//
// Package:    HeavyIonAnalyzer/ZDCAnalysis/FSCAnalyzerHC
// Class:      FSCAnalyzerHC
//
/**\class FSCAnalyzerHC FSCAnalyzerHC.cc HeavyIonAnalyzer/ZDCAnalysis/plugins/FSCAnalyzerHC

   Description: Produced Tree with ZDC RecHit and zdcdigi information 
*/
//
// Original Author:  Matthew Nickel, University of Kansas
//         Created:  23-06-2025
//         Modified from ZDCRecHitAnalyzerHC
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"

#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

#include "FSCstruct.h"
#include "ZDCHardCodeHelper.h"

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

using reco::TrackCollection;

class FSCAnalyzerHC : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit FSCAnalyzerHC(const edm::ParameterSet&);
  ~FSCAnalyzerHC();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<QIE10DigiCollection> ZDCDigiToken_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> hcalDatabaseToken_;
  bool doFullFitFSC_;
  bool do50nsRecoFSC_;
  bool doHardcodedFSC_;
  edm::Service<TFileService> fs;
  TTree* t1;

  MyFSCDigi fscDigi;

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  edm::ESGetToken<SetupData, SetupRecord> setupToken_;
#endif
};

//
// constants, enums and typedefs
//
// This information is temrorarly stored here, once the sequence is validated it will be ported to the database
// These numbers extracted on November 16, 2025 (run=399551). Any time timing calibration is applied, the F3, F4, F5 need to be updated
namespace {
  // 2 sides (−, +) × NFSC channels
  constexpr float Pedestal[2][NFSC] = {
    {1248.54f, 1287.14f, 1129.79f, 1216.63f, 1285.63f, 1170.79f}, // side minus
    {1202.87f, 1199.81f, 1187.31f, 1218.55f, 1147.67f, 1231.52f}, // side plus
  };

  constexpr float f3[2][NFSC] = {
    {0.485614f, 0.37147f, 0.431836f, 0.357515f, 0.432989f, 0.314054f}, // side minus
    {0.40573f, 0.368565f, 0.239196f, 0.281138f, 0.295968f, 0.290039f}, // side plus
  };

  constexpr float f4[2][NFSC] = {
    {0.191546f, 0.159452f, 0.142411f, 0.131297f, 0.154015f, 0.11559f}, // side minus
    {0.145969f, 0.134359f, 0.0892946f, 0.101017f, 0.101158f, 0.1043f}, // side plus
  };

  constexpr float f5[2][NFSC] = {
    {0.119143f, 0.0990354f, 0.080384f, 0.0701533f, 0.0912764f, 0.0629495f}, // side minus
    {0.083688f, 0.0764629f, 0.0522635f, 0.0588876f, 0.0572844f, 0.0595602f}, // side plus
  };

  // pO pulse shape constants, uncomment in case if running on oxygen data (extracted from run=393953)
  //constexpr float Pedestal[2][NFSC] = {{1240.81f, 1244.26f, 1129.9f, 1217.23f, 1279.97f, 1169.07f},{}};
  //constexpr float f3[2][NFSC] = {{0.432071f, 0.37209f, 0.66697f, 0.679988f, 0.80187f, 0.483295f},{}};
  //constexpr float f4[2][NFSC] = {{0.175788f, 0.159102f, 0.1914f, 0.199924f, 0.233085f, 0.150607f}, {}};
  //constexpr float f5[2][NFSC] = {{0.110565f, 0.0990709f, 0.0991153f, 0.102377f, 0.129583f, 0.079309f},{}};
}  // namespace
//
// static data member definitions
//

//
// constructors and destructor
//
FSCAnalyzerHC::FSCAnalyzerHC(const edm::ParameterSet& iConfig)
    : ZDCDigiToken_(consumes<QIE10DigiCollection>(iConfig.getParameter<edm::InputTag>("ZDCDigiSource"))),
      hcalDatabaseToken_(esConsumes<HcalDbService, HcalDbRecord>()),
      doFullFitFSC_(iConfig.getParameter<bool>("doFullFitFSC")),
      do50nsRecoFSC_(iConfig.getParameter<bool>("do50nsRecoFSC")),
      doHardcodedFSC_(iConfig.getParameter<bool>("doHardcodedFSC")) {
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  setupDataToken_ = esConsumes<SetupData, SetupRecord>();
#endif
    // ---- PROTECTION: user must choose exactly one method ----
    int nSelected = (doFullFitFSC_ ? 1 : 0) + (do50nsRecoFSC_ ? 1 : 0);

    if (nSelected != 1) {
        throw cms::Exception("Configuration")
            << "Invalid FSC charge reconstruction configuration:\n"
            << "  doFullFitFSC = " << doFullFitFSC_ << "\n"
            << "  do50nsRecoFSC = " << do50nsRecoFSC_ << "\n"
            << "Exactly ONE of these must be set to True.\n"
            << "Please fix the configuration and rerun.\n";
    }

    // now continue with other initialization…
}


FSCAnalyzerHC::~FSCAnalyzerHC() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called for each event  ------------
void FSCAnalyzerHC::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  ZDCHardCodeHelper HardCodeZDC;
  edm::Handle<QIE10DigiCollection> zdcdigis;

  iEvent.getByToken(ZDCDigiToken_, zdcdigis);

  edm::ESHandle<HcalDbService> conditions = iSetup.getHandle(hcalDatabaseToken_);

  fscDigi.sumPlus = 0;
  fscDigi.sumMinus = 0;
  fscDigi.sumPlus_FSC2only = 0;
  fscDigi.sumPlus_FSC3only = 0;
  fscDigi.sumMinus_FSC2only = 0;
  fscDigi.sumMinus_FSC3only = 0;
  
  fscDigi.n = 0;
  for (unsigned int i = 0; i < NMOD; i++) {
    fscDigi.zside[i] = -99;
    fscDigi.section[i] = -99;
    fscDigi.channel[i] = -99;
    for (int ts = 0; ts < NTS; ts++) {
      fscDigi.chargefC[ts][i] = -99;
      fscDigi.adc[ts][i] = -99;
      fscDigi.tdc[ts][i] = -99;
    }
    fscDigi.charge[i] = -99;
    fscDigi.charge_bare[i] = -99;
    fscDigi.Fitted_QTS0[i] = -99;
    fscDigi.Fitted_QTS2[i] = -99;
    fscDigi.saturation[i] = -99;
  }

  int nhits = 0;
  for (auto it = zdcdigis->begin(); it != zdcdigis->end(); it++) {
    const QIE10DataFrame digi = static_cast<const QIE10DataFrame>(*it);

    HcalZDCDetId zdcid = digi.id();
    int zside = zdcid.zside();
    int section = zdcid.section();
    int channel = zdcid.channel();

    if (nhits >= NMOD)
      break;
    if (!(section == 1 && channel > 6))
      continue;  // Only consider FSC channel located in dummy ECAL

    CaloSamples caldigi;

    //const ZDCDataFrame & rh = (*zdcdigis)[it];

    if (!(doHardcodedFSC_)) {
      const HcalQIECoder* qiecoder = conditions->getHcalCoder(zdcid);
      const HcalQIEShape* qieshape = conditions->getHcalShape(qiecoder);
      HcalCoderDb coder(*qiecoder, *qieshape);
      // coder.adc2fC(rh,caldigi);
      coder.adc2fC(digi, caldigi);
    }

    fscDigi.zside[nhits] = zside;
    fscDigi.section[nhits] = section;
    fscDigi.channel[nhits] = channel;

    for (int ts = 0; ts < digi.samples(); ts++) {
      fscDigi.adc[ts][nhits] = digi[ts].adc();
      fscDigi.tdc[ts][nhits] = digi[ts].le_tdc();
      if (doHardcodedFSC_) {
        fscDigi.chargefC[ts][nhits] = HardCodeZDC.charge(digi[ts].adc(), digi[ts].capid());
      } else {
        fscDigi.chargefC[ts][nhits] = caldigi[ts];
      }
    }
	
	// CHARGE RECONSTRUCTION
	int side = (zside<0) ? 0 : 1;
	int ch = channel - 7;
	float ped = Pedestal[side][ch];
	float f3v=f3[side][ch];
	float f4v=f4[side][ch];
	float f5v=f5[side][ch];
	int saturation = 0;

	float Q0 = fscDigi.chargefC[0][nhits] - ped, Q1 = fscDigi.chargefC[1][nhits] - ped;
	float Q2 = fscDigi.chargefC[2][nhits] - ped, Q3 = fscDigi.chargefC[3][nhits] - ped;
	float Q4 = fscDigi.chargefC[4][nhits] - ped, Q5 = fscDigi.chargefC[5][nhits] - ped;
	float charge = 0;
	
	// Reconstruct the charge in Q2
	if (do50nsRecoFSC_){ // apply correction from the first time slice

        auto chi2 = [&](double A, double B) {
            double r0 = B + A * f4v - Q0;
            double r1 = B * f3v + A * Q1;
            return r0 * r0 + r1 * r1;
        };

        double bestB    = 0.0;
        double bestChi2 = std::numeric_limits<double>::infinity();

        // --- 0) Unconstrained interior solution ---
        double denom = f3v * f4v - f5v;
        if (std::fabs(denom) > 1e-12) {
            double A_star = (Q0 * f3v - Q1) / denom;
            double B_star = (Q1 * f4v - Q0 * f5v) / denom;
            if (A_star >= 0.0 && B_star >= 0.0) {
                double c = chi2(A_star, B_star);
                bestChi2 = c;
                bestB    = B_star;
            }
        }		

        // --- 1) Boundary case A = 0 ---
        {
            double denomA = 1.0 + f3v * f3v;
            if (denomA > 1e-12) {
                double B_A = (Q0 + f3v * Q1) / denomA;
                if (B_A >= 0.0) {
                    double c = chi2(0.0, B_A);
                    if (c < bestChi2) {
                        bestChi2 = c;
                        bestB    = B_A;
                    }
                }
            }
        }
        // --- 2) Boundary case B = 0 ---
        {
            double denomB = f4v * f4v + f5v * f5v;
            if (denomB > 1e-12) {
                double A_B = (Q0 * f4v + Q1 * f5v) / denomB;
                if (A_B >= 0.0) {
                    double c = chi2(A_B, 0.0);
                    if (c < bestChi2) {
                        bestChi2 = c;
                        bestB    = 0.0;
                    }
                }
            }
        }
        // --- 3) Corner A = 0, B = 0 ---
        {
            double c = chi2(0.0, 0.0);
            if (c < bestChi2) {
                bestChi2 = c;
                bestB    = 0.0;
            }
        }	

        // Best-fit previous-bunch charge
        Q0 = static_cast<float>(bestB);	
	}
	else{ Q0 = 0; }
		
	if (doFullFitFSC_) {

		// ===== Full fit: TS2–TS5 optimal combination =====

		if (fscDigi.adc[2][nhits] == 255) {          // TS2 saturated
			if (fscDigi.adc[3][nhits] == 255) {      // TS3 saturated
				if (fscDigi.adc[4][nhits] == 255) {  // TS4 saturated
					// TS2–TS4 saturated → only TS5 available
					Q2 = Q5 / f5v;

					if (fscDigi.adc[5][nhits] == 255) { // TS5 saturated
						saturation = 5;                 // TS2–TS5 saturated
					} else {
						saturation = 4;                 // TS2–TS4 saturated
					}
				}
				else {
					// TS2–TS3 saturated, TS4 OK → use TS4 & TS5
					Q2 = (f4v * Q4 + f5v * Q5) / (f4v*f4v + f5v*f5v);
					saturation = 3;
				}
			}
			else {
				// TS2 saturated, TS3 OK → use TS3–TS5
				Q2 = (f3v * Q3 + f4v * Q4 + f5v * Q5) /
					 (f3v*f3v + f4v*f4v + f5v*f5v);
				saturation = 1;
			}
		}
		else {
			// TS2 not saturated → full optimal combination
			Q2 = (Q2 + f3v * Q3 + f4v * Q4 + f5v * Q5) /
				 (1.0f + f3v*f3v + f4v*f4v + f5v*f5v);
			saturation = 0;
		}

	} else {

		// subtract signal leackage from out-of-time pileup signal
		Q2 -= Q0 * f4v;
		Q3 -= Q0 * f5v;
		
		// ===== Reduced fit: ONLY TS2 and TS3  =====

		if (fscDigi.adc[2][nhits] == 255) {   // TS2 saturated
			Q2 = Q3 / f3v;

			if (fscDigi.adc[3][nhits] == 255) { // TS3 saturated
				saturation = 2;
			}
			else {
				saturation = 1;
			}
		}
		else {
			Q2 = (Q2 + f3v * Q3) / (1.0f + f3v*f3v);
			saturation = 0;
		}
	}


	if(fscDigi.adc[0][nhits]==255) saturation +=10; // TS0 saturated
	
	// new calibrated charge
	charge = Q2 * (1.0f + f3v + f4v + f5v);
	
	fscDigi.Fitted_QTS0[nhits] = Q0 + ped;
	fscDigi.Fitted_QTS2[nhits] = Q2 + ped;
	fscDigi.saturation[nhits] = saturation;
	fscDigi.charge[nhits] = charge;	
	fscDigi.charge_bare[nhits] = (fscDigi.chargefC[2][nhits] - ped) * (1.0f + f3v + f4v + f5v);
	
    nhits++;
	
	// add sums:
	if(zside < 0){
		fscDigi.sumMinus += charge;
		if(channel < 9) fscDigi.sumMinus_FSC2only += charge;
		else fscDigi.sumMinus_FSC3only += charge;
	}
	else{
		fscDigi.sumPlus += charge;
		if(channel < 9) fscDigi.sumPlus_FSC2only += charge;
		else fscDigi.sumPlus_FSC3only += charge;
	}
	
  }  // end loop zdc digis

  fscDigi.n = nhits;

  t1->Fill();

  // #ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  // // if the SetupData is always needed
  // auto setup = iSetup.getData(setupToken_);
  // // if need the ESHandle to check if the SetupData was there or not
  // auto pSetup = iSetup.getHandle(setupToken_);
  // #endif
}

// ------------ method called once each job just before starting event loop  ------------
void FSCAnalyzerHC::beginJob() {
  t1 = fs->make<TTree>("fscdigi", "fscdigi");

  t1->Branch("n", &fscDigi.n, "n/I");
  t1->Branch("zside", fscDigi.zside, "zside[n]/I");
  t1->Branch("section", fscDigi.section, "section[n]/I");
  t1->Branch("channel", fscDigi.channel, "channel[n]/I");

  for (int i = 0; i < NTS; i++) {
    TString adcTsSt("adcTs"), chargefCTsSt("chargefCTs"), tdcTsSt("tdcTs");
    adcTsSt += i;
    chargefCTsSt += i;
    tdcTsSt += i;

    t1->Branch(adcTsSt, fscDigi.adc[i], adcTsSt + "[n]/I");
    t1->Branch(chargefCTsSt, fscDigi.chargefC[i], chargefCTsSt + "[n]/F");
    t1->Branch(tdcTsSt, fscDigi.tdc[i], tdcTsSt + "[n]/I");
  }

  t1->Branch("Fitted_QTS0", fscDigi.Fitted_QTS0, "Fitted_QTS0[n]/F");
  t1->Branch("Fitted_QTS2", fscDigi.Fitted_QTS2, "Fitted_QTS2[n]/F");
  t1->Branch("saturation", fscDigi.saturation, "saturation[n]/I");
  t1->Branch("charge", fscDigi.charge, "charge[n]/F");
  t1->Branch("charge_bare", fscDigi.charge_bare, "charge_bare[n]/F");
  t1->Branch("sumMinus", &fscDigi.sumMinus);
  t1->Branch("sumMinus_FSC2only", &fscDigi.sumMinus_FSC2only);
  t1->Branch("sumMinus_FSC3only", &fscDigi.sumMinus_FSC3only);
  t1->Branch("sumPlus", &fscDigi.sumPlus);
  t1->Branch("sumPlus_FSC2only", &fscDigi.sumPlus_FSC2only);
  t1->Branch("sumPlus_FSC3only", &fscDigi.sumPlus_FSC3only);
}

// ------------ method called once each job just after ending the event loop  ------------
void FSCAnalyzerHC::endJob() {
  // please remove this method if not needed
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void FSCAnalyzerHC::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(FSCAnalyzerHC);
