//!
//! This piece of code is obsolete and completely unused. It is kept as an
//! example to show how to access certain geometry information.
//!

#include <cassert>
#include <cmath>
#include <memory>
#include <vector>
#include <fstream>
#include <iostream>

#include "TFile.h"
#include "TTree.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

//#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"

#include "L1Trigger/L1TMuon/interface/GeometryTranslator.h"
#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitive.h"
#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitiveFwd.h"

typedef L1TMuon::GeometryTranslator GeometryTranslator;
typedef L1TMuon::TriggerPrimitive TriggerPrimitive;
typedef L1TMuon::TriggerPrimitiveCollection TriggerPrimitiveCollection;

#include "helper.h"

class MakeAngleLUT : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit MakeAngleLUT(const edm::ParameterSet&);
  ~MakeAngleLUT() override;

private:
  //virtual void beginJob();
  //virtual void endJob();

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  void generateLUTs();

private:
  GeometryTranslator geometry_translator_;

  const edm::ParameterSet config_;

  int verbose_;

  std::string outfile_;

  bool done_;

  /// Event setup
};

// _____________________________________________________________________________
MakeAngleLUT::MakeAngleLUT(const edm::ParameterSet& iConfig)
    : geometry_translator_(consumesCollector()),
      config_(iConfig),
      verbose_(iConfig.getUntrackedParameter<int>("verbosity")),
      outfile_(iConfig.getParameter<std::string>("outfile")),
      done_(false) {
  assert(CSCConstants::KEY_CLCT_LAYER == CSCConstants::KEY_ALCT_LAYER);
}

MakeAngleLUT::~MakeAngleLUT() {}

void MakeAngleLUT::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  geometry_translator_.checkAndUpdateGeometry(iSetup);
}

void MakeAngleLUT::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {}

void MakeAngleLUT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (done_)
    return;

  generateLUTs();

  done_ = true;
  return;
}

// _____________________________________________________________________________
void MakeAngleLUT::generateLUTs() {
  const CSCGeometry& geocsc = geometry_translator_.getCSCGeometry();
  const RPCGeometry& georpc = geometry_translator_.getRPCGeometry();
  const GEMGeometry& geogem = geometry_translator_.getGEMGeometry();
  const MagneticField& magfield = geometry_translator_.getMagneticField();

  auto average = [](double x, double y) { return 0.5 * (x + y); };

  // Save z positions for ME1/1, ME1/2, ME1/3, ME2/2, ME3/2, ME4/2,
  //                      RE1/2, RE1/3, RE2/2, RE3/2, RE4/2,
  //                      GE1/1, GE1/2,
  std::vector<double> z_positions(13 * 2, 0.);

  // ___________________________________________________________________________
  // CSC

  for (const auto& it : geocsc.detUnits()) {
    const CSCLayer* layer = dynamic_cast<const CSCLayer*>(it);  // like GeomDetUnit
    assert(layer != nullptr);
    const CSCChamber* chamber = layer->chamber();  // like GeomDet
    assert(chamber != nullptr);
    const CSCDetId& cscDetId = chamber->id();
    //double zpos = chamber->surface().position().z();  // [cm]
    double zpos = chamber->layer(CSCConstants::KEY_ALCT_LAYER)->surface().position().z();  // [cm]
    //std::cout << "CSC: " << cscDetId.endcap() << " " << cscDetId.station() << " " << cscDetId.ring() << " " << cscDetId.chamber() << " " << cscDetId.layer() << " " << zpos << std::endl;

    // Save the numbers
    if (cscDetId.endcap() == 1 && (cscDetId.chamber() == 1 || cscDetId.chamber() == 2)) {
      if (cscDetId.station() == 1 && cscDetId.ring() == 1) {
        if (cscDetId.chamber() == 2) {  // front
          z_positions[0] = zpos;
        } else if (cscDetId.chamber() == 1) {  // rear
          z_positions[1] = zpos;
        }
      } else if (cscDetId.station() == 1 && cscDetId.ring() == 2) {
        if (cscDetId.chamber() == 2) {  // front
          z_positions[2] = zpos;
        } else if (cscDetId.chamber() == 1) {  // rear
          z_positions[3] = zpos;
        }
      } else if (cscDetId.station() == 1 && cscDetId.ring() == 3) {
        if (cscDetId.chamber() == 2) {  // front
          z_positions[4] = zpos;
        } else if (cscDetId.chamber() == 1) {  // rear
          z_positions[5] = zpos;
        }
      } else if (cscDetId.station() == 2 && cscDetId.ring() == 2) {
        if (cscDetId.chamber() == 2) {  // front
          z_positions[6] = zpos;
        } else if (cscDetId.chamber() == 1) {  // rear
          z_positions[7] = zpos;
        }
      } else if (cscDetId.station() == 3 && cscDetId.ring() == 2) {
        if (cscDetId.chamber() == 1) {  // front
          z_positions[8] = zpos;
        } else if (cscDetId.chamber() == 2) {  // rear
          z_positions[9] = zpos;
        }
      } else if (cscDetId.station() == 4 && cscDetId.ring() == 2) {
        if (cscDetId.chamber() == 1) {  // front
          z_positions[10] = zpos;
        } else if (cscDetId.chamber() == 2) {  // rear
          z_positions[11] = zpos;
        }
      }
    }
  }  // end loop over CSC detUnits

  // ___________________________________________________________________________
  // RPC

  for (const auto& it : georpc.detUnits()) {
    const RPCRoll* roll = dynamic_cast<const RPCRoll*>(it);  // like GeomDetUnit
    assert(roll != nullptr);
    //const RPCChamber* chamber = roll->chamber();  // like GeomDet
    //assert(chamber != nullptr);
    const RPCDetId& rpcDetId = roll->id();
    if (rpcDetId.region() == 0)  // skip barrel
      continue;
    //if (rpcDetId.region() == 0 || (rpcDetId.station() <= 2 && rpcDetId.ring() == 3))  // skip barrel, RE1/3, RE2/3
    //  continue;
    double zpos = roll->surface().position().z();  // [cm]
    //std::cout << "RPC: " << rpcDetId.region() << " " << rpcDetId.ring() << " " << rpcDetId.station() << " " << rpcDetId.sector() << " " << rpcDetId.layer() << " " << rpcDetId.subsector() << " " << rpcDetId.roll() << " " << zpos << std::endl;

    // Save the numbers
    if (rpcDetId.region() == 1 && rpcDetId.sector() == 1 && rpcDetId.roll() == 1 &&
        (rpcDetId.subsector() == 1 || rpcDetId.subsector() == 2)) {
      if (rpcDetId.station() == 1 && rpcDetId.ring() == 2) {
        if (rpcDetId.subsector() == 2) {  // front
          z_positions[12] = zpos;
        } else if (rpcDetId.subsector() == 1) {  // rear
          z_positions[13] = zpos;
        }
      } else if (rpcDetId.station() == 1 && rpcDetId.ring() == 3) {
        if (rpcDetId.subsector() == 2) {  // front
          z_positions[14] = zpos;
        } else if (rpcDetId.subsector() == 1) {  // rear
          z_positions[15] = zpos;
        }
      } else if (rpcDetId.station() == 2 && rpcDetId.ring() == 2) {
        if (rpcDetId.subsector() == 2) {  // front
          z_positions[16] = zpos;
        } else if (rpcDetId.subsector() == 1) {  // rear
          z_positions[17] = zpos;
        }
      } else if (rpcDetId.station() == 3 && rpcDetId.ring() == 2) {
        if (rpcDetId.subsector() == 2) {  // front
          z_positions[18] = zpos;
        } else if (rpcDetId.subsector() == 1) {  // rear
          z_positions[19] = zpos;
        }
      } else if (rpcDetId.station() == 4 && rpcDetId.ring() == 2) {
        if (rpcDetId.subsector() == 2) {  // front
          z_positions[20] = zpos;
        } else if (rpcDetId.subsector() == 1) {  // rear
          z_positions[21] = zpos;
        }
      }
    }
  }  // end loop over RPC detUnits

  // ___________________________________________________________________________
  // GEM

  for (const auto& it : geogem.detUnits()) {
    const GEMEtaPartition* roll = dynamic_cast<const GEMEtaPartition*>(it);  // like GeomDetUnit
    assert(roll != nullptr);
    //const GEMChamber* chamber = roll->chamber();  // like GeomDet
    //assert(chamber != nullptr);
    const GEMDetId& gemDetId = roll->id();
    if (gemDetId.region() == 0)  // skip barrel
      continue;
    //double zpos = roll->surface().position().z();  // [cm]
    //std::cout << "GEM: " << gemDetId.region() << " " << gemDetId.ring() << " " << gemDetId.station() << " " << gemDetId.layer() << " " << gemDetId.chamber() << " " << gemDetId.roll() << " " << zpos << std::endl;
  }  // end loop over GEM detUnits

  // ___________________________________________________________________________
  // Verbose

  if (verbose_) {
    std::cout << "z positions:" << std::endl;
    for (const auto& zpos : z_positions) {
      std::cout << zpos << std::endl;
    }
    std::cout << std::endl;

    std::cout << "common planes:" << std::endl;
    std::cout << average(z_positions[0], z_positions[1]) << std::endl;
    std::cout << average(z_positions[2], z_positions[3]) << std::endl;
    std::cout << average(z_positions[6], z_positions[7]) << std::endl;
    std::cout << average(z_positions[8], z_positions[9]) << std::endl;
    std::cout << average(z_positions[10], z_positions[11]) << std::endl;
    std::cout << std::endl;
  }

  // Calculate the coefficients
  auto get_eta_bin_center = [](int b) {
    // nbinsx, xlow, xup = 2048, 1.1, 2.5
    double c = 1.1 + (2.5 - 1.1) / 2048. * (0.5 + static_cast<double>(b));
    return c;
  };

  int num_z_bins = z_positions.size();
  int num_eta_bins = 2048;

  std::vector<double> coefficients;

  // Loop over z bins
  for (int iz = 0; iz < num_z_bins; ++iz) {
    // Find common plane
    double common_zpos = 0.;
    if (iz == 0 || iz == 1) {  // ME1/1
      common_zpos = average(z_positions[0], z_positions[1]);
    } else if (iz == 2 || iz == 3 || iz == 4 || iz == 5) {  // ME1/2, ME1/3
      common_zpos = average(z_positions[2], z_positions[3]);
    } else if (iz == 6 || iz == 7) {  // ME2/2
      common_zpos = average(z_positions[6], z_positions[7]);
    } else if (iz == 8 || iz == 9) {  // ME3/2
      common_zpos = average(z_positions[8], z_positions[9]);
    } else if (iz == 10 || iz == 11) {  // ME4/2
      common_zpos = average(z_positions[10], z_positions[11]);
    } else if (iz == 12 || iz == 13 || iz == 14 || iz == 15) {  // RE1/2, RE1/3
      common_zpos = average(z_positions[2], z_positions[3]);
    } else if (iz == 16 || iz == 17) {  // RE2/2
      common_zpos = average(z_positions[6], z_positions[7]);
    } else if (iz == 18 || iz == 19) {  // RE3/2
      common_zpos = average(z_positions[8], z_positions[9]);
    } else if (iz == 20 || iz == 21) {  // RE4/2
      common_zpos = average(z_positions[10], z_positions[11]);
    }

    // Loop over eta bins
    for (int ieta = 0; ieta < num_eta_bins; ++ieta) {
      // Find magnetic field strength
      double zpos = z_positions.at(iz);
      double deltaZ = zpos - common_zpos;
      double eta = get_eta_bin_center(ieta);
      double cotTheta = std::sinh(eta);
      double r = zpos / cotTheta;

      const GlobalPoint gp(r, 0, zpos);      // phi = 0
      double bz = magfield.inTesla(gp).z();  // [Tesla]

      // Calculate the coefficient
      // coeff = deltaZ * (-0.5) * 0.003 * bz
      double coeff = deltaZ * (-0.5) * 0.003 * bz;
      coefficients.push_back(coeff);
    }
  }
  assert(coefficients.size() == (unsigned)(num_z_bins * num_eta_bins));

  // Write
  {
    TFile* tfile = TFile::Open(outfile_.c_str(), "RECREATE");
    TTree* ttree = new TTree("tree", "tree");
    ttree->Branch("coefficients", &coefficients);
    ttree->Fill();
    tfile->Write();
    std::cout << "Wrote file: " << outfile_ << std::endl;
  }
}

// DEFINE THIS AS A PLUG-IN
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MakeAngleLUT);
