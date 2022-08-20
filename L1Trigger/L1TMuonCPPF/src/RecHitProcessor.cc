#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "L1Trigger/L1TMuonCPPF/interface/RecHitProcessor.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "L1Trigger/L1TMuonCPPF/src/CPPFClusterContainer.h"
#include "L1Trigger/L1TMuonCPPF/src/CPPFCluster.h"
#include "L1Trigger/L1TMuonCPPF/src/CPPFClusterizer.h"
#include "L1Trigger/L1TMuonCPPF/src/CPPFMaskReClusterizer.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

RecHitProcessor::RecHitProcessor() {}

RecHitProcessor::~RecHitProcessor() {}

void RecHitProcessor::processLook(const edm::Event &iEvent,
                                  const edm::EventSetup &iSetup,
                                  const edm::EDGetToken &recHitToken,
                                  const edm::EDGetToken &rpcDigiToken,
                                  const edm::EDGetToken &rpcDigiSimLinkToken,
                                  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> &rpcGeomToken,
                                  std::vector<RecHitProcessor::CppfItem> &CppfVec1,
                                  l1t::CPPFDigiCollection &cppfDigis,
                                  const int MaxClusterSize) const {
  edm::Handle<RPCRecHitCollection> recHits;
  iEvent.getByToken(recHitToken, recHits);

  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByToken(rpcDigiToken, rpcDigis);

  edm::Handle<edm::DetSetVector<RPCDigiSimLink>> theSimlinkDigis;
  iEvent.getByToken(rpcDigiSimLinkToken, theSimlinkDigis);

  const auto &rpcGeom = iSetup.getData(rpcGeomToken);

  for (const auto &&rpcdgIt : (*rpcDigis)) {
    const RPCDetId &rpcId = rpcdgIt.first;
    const RPCDigiCollection::Range &range = rpcdgIt.second;
    if (rpcId.region() == 0)
      continue;

    CPPFClusterizer clizer;
    CPPFClusterContainer tcls = clizer.doAction(range);
    CPPFMaskReClusterizer mrclizer;
    CPPFRollMask mask;
    CPPFClusterContainer cls = mrclizer.doAction(rpcId, tcls, mask);

    for (const auto &cl : cls) {
      int isValid = rpcDigis.isValid();
      int rawId = rpcId.rawId();
      int Bx = cl.bx();
      const int firststrip = cl.firstStrip();
      const int clustersize = cl.clusterSize();
      const int laststrip = cl.lastStrip();
      const RPCRoll *roll = rpcGeom.roll(rpcId);
      // Get Average Strip position
      const float fstrip = (roll->centreOfStrip(firststrip)).x();
      const float lstrip = (roll->centreOfStrip(laststrip)).x();
      const float centreOfCluster = (fstrip + lstrip) / 2;
      const double y = cl.hasY() ? cl.y() : 0;
      LocalPoint lPos(centreOfCluster, y, 0);
      if (roll->id().region() != 0) {
        const auto &topo = dynamic_cast<const TrapezoidalStripTopology &>(roll->topology());
        const double angle = topo.stripAngle((firststrip + laststrip) / 2.);
        const double x = centreOfCluster - y * std::tan(angle);
        lPos = LocalPoint(x, y, 0);
      }
      const BoundPlane &rollSurface = roll->surface();
      GlobalPoint gPos = rollSurface.toGlobal(lPos);
      float global_theta = emtf::rad_to_deg(gPos.theta().value());
      float global_phi = emtf::rad_to_deg(gPos.phi().value());

      // Establish the average position of the rechit
      int rechitstrip = firststrip;

      if (clustersize > 2) {
        int medium = 0;
        if (clustersize % 2 == 0)
          medium = 0.5 * (clustersize);
        else
          medium = 0.5 * (clustersize - 1);
        rechitstrip += medium;
      }
      if (clustersize > MaxClusterSize)
        continue;
      // This is just for test CPPFDigis with the RPC Geometry, It must be
      // "true" in the normal runs
      bool Geo = true;
      ////:::::::::::::::::::::::::::::::::::::::::::::::::
      // Set the EMTF Sector
      int EMTFsector1 = 0;
      int EMTFsector2 = 0;

      // sector 1
      if ((global_phi > 15.) && (global_phi <= 16.3)) {
        EMTFsector1 = 1;
        EMTFsector2 = 6;
      } else if ((global_phi > 16.3) && (global_phi <= 53.)) {
        EMTFsector1 = 1;
        EMTFsector2 = 0;
      } else if ((global_phi > 53.) && (global_phi <= 75.)) {
        EMTFsector1 = 1;
        EMTFsector2 = 2;
      }
      // sector 2
      else if ((global_phi > 75.) && (global_phi <= 76.3)) {
        EMTFsector1 = 1;
        EMTFsector2 = 2;
      } else if ((global_phi > 76.3) && (global_phi <= 113.)) {
        EMTFsector1 = 2;
        EMTFsector2 = 0;
      } else if ((global_phi > 113.) && (global_phi <= 135.)) {
        EMTFsector1 = 2;
        EMTFsector2 = 3;
      }
      // sector 3
      // less than 180
      else if ((global_phi > 135.) && (global_phi <= 136.3)) {
        EMTFsector1 = 2;
        EMTFsector2 = 3;
      } else if ((global_phi > 136.3) && (global_phi <= 173.)) {
        EMTFsector1 = 3;
        EMTFsector2 = 0;
      } else if ((global_phi > 173.) && (global_phi <= 180.)) {
        EMTFsector1 = 3;
        EMTFsector2 = 4;
      }
      // Greater than -180
      else if ((global_phi < -165.) && (global_phi >= -180.)) {
        EMTFsector1 = 3;
        EMTFsector2 = 4;
      }
      // Fourth sector
      else if ((global_phi > -165.) && (global_phi <= -163.7)) {
        EMTFsector1 = 3;
        EMTFsector2 = 4;
      } else if ((global_phi > -163.7) && (global_phi <= -127.)) {
        EMTFsector1 = 4;
        EMTFsector2 = 0;
      } else if ((global_phi > -127.) && (global_phi <= -105.)) {
        EMTFsector1 = 4;
        EMTFsector2 = 5;
      }
      // fifth sector
      else if ((global_phi > -105.) && (global_phi <= -103.7)) {
        EMTFsector1 = 4;
        EMTFsector2 = 5;
      } else if ((global_phi > -103.7) && (global_phi <= -67.)) {
        EMTFsector1 = 5;
        EMTFsector2 = 0;
      } else if ((global_phi > -67.) && (global_phi <= -45.)) {
        EMTFsector1 = 5;
        EMTFsector2 = 6;
      }
      // sixth sector
      else if ((global_phi > -45.) && (global_phi <= -43.7)) {
        EMTFsector1 = 5;
        EMTFsector2 = 6;
      } else if ((global_phi > -43.7) && (global_phi <= -7.)) {
        EMTFsector1 = 6;
        EMTFsector2 = 0;
      } else if ((global_phi > -7.) && (global_phi <= 15.)) {
        EMTFsector1 = 6;
        EMTFsector2 = 1;
      }

      double EMTFLink1 = 0.;
      double EMTFLink2 = 0.;
      std::vector<RecHitProcessor::CppfItem>::iterator cppf1;
      std::vector<RecHitProcessor::CppfItem>::iterator cppf;
      for (cppf1 = CppfVec1.begin(); cppf1 != CppfVec1.end(); cppf1++) {
        // Condition to save the CPPFDigi
        if (((*cppf1).rawId == rawId) && ((*cppf1).strip == rechitstrip)) {
          int old_strip = (*cppf1).strip;
          int before = 0;
          int after = 0;

          if (cppf1 != CppfVec1.begin())
            before = (*(cppf1 - 2)).strip;
          else if (cppf1 == CppfVec1.begin())
            before = (*cppf1).strip;
          if (cppf1 != CppfVec1.end())
            after = (*(cppf1 + 2)).strip;
          else if (cppf1 == CppfVec1.end())
            after = (*cppf1).strip;
          cppf = cppf1;

          if (clustersize == 2) {
            if (firststrip == 1) {
              if (before < after)
                cppf = (cppf1 - 1);
              else if (before > after)
                cppf = (cppf1 + 1);
            } else if (firststrip > 1) {
              if (before < after)
                cppf = (cppf1 + 1);
              else if (before > after)
                cppf = (cppf1 - 1);
            }
          }
          // Using the RPCGeometry
          if (Geo) {
            std::shared_ptr<l1t::CPPFDigi> MainVariables1(new l1t::CPPFDigi(rpcId,
                                                                            Bx,
                                                                            (*cppf).int_phi,
                                                                            (*cppf).int_theta,
                                                                            isValid,
                                                                            (*cppf).lb,
                                                                            (*cppf).halfchannel,
                                                                            EMTFsector1,
                                                                            EMTFLink1,
                                                                            old_strip,
                                                                            clustersize,
                                                                            global_phi,
                                                                            global_theta));
            std::shared_ptr<l1t::CPPFDigi> MainVariables2(new l1t::CPPFDigi(rpcId,
                                                                            Bx,
                                                                            (*cppf).int_phi,
                                                                            (*cppf).int_theta,
                                                                            isValid,
                                                                            (*cppf).lb,
                                                                            (*cppf).halfchannel,
                                                                            EMTFsector2,
                                                                            EMTFLink2,
                                                                            old_strip,
                                                                            clustersize,
                                                                            global_phi,
                                                                            global_theta));

            if ((EMTFsector1 > 0) && (EMTFsector2 == 0)) {
              cppfDigis.push_back(*MainVariables1.get());
            } else if ((EMTFsector1 > 0) && (EMTFsector2 > 0)) {
              cppfDigis.push_back(*MainVariables1.get());
              cppfDigis.push_back(*MainVariables2.get());
            } else if ((EMTFsector1 == 0) && (EMTFsector2 == 0)) {
              continue;
            }
          }  // Geo is true
          else {
            global_phi = 0.;
            global_theta = 0.;
            std::shared_ptr<l1t::CPPFDigi> MainVariables1(new l1t::CPPFDigi(rpcId,
                                                                            Bx,
                                                                            (*cppf).int_phi,
                                                                            (*cppf).int_theta,
                                                                            isValid,
                                                                            (*cppf).lb,
                                                                            (*cppf).halfchannel,
                                                                            EMTFsector1,
                                                                            EMTFLink1,
                                                                            old_strip,
                                                                            clustersize,
                                                                            global_phi,
                                                                            global_theta));
            std::shared_ptr<l1t::CPPFDigi> MainVariables2(new l1t::CPPFDigi(rpcId,
                                                                            Bx,
                                                                            (*cppf).int_phi,
                                                                            (*cppf).int_theta,
                                                                            isValid,
                                                                            (*cppf).lb,
                                                                            (*cppf).halfchannel,
                                                                            EMTFsector2,
                                                                            EMTFLink2,
                                                                            old_strip,
                                                                            clustersize,
                                                                            global_phi,
                                                                            global_theta));
            if ((EMTFsector1 > 0) && (EMTFsector2 == 0)) {
              cppfDigis.push_back(*MainVariables1.get());
            } else if ((EMTFsector1 > 0) && (EMTFsector2 > 0)) {
              cppfDigis.push_back(*MainVariables1.get());
              cppfDigis.push_back(*MainVariables2.get());
            } else if ((EMTFsector1 == 0) && (EMTFsector2 == 0)) {
              continue;
            }
          }
        }  // Condition to save the CPPFDigi
      }    // Loop over the LUTVector
    }      //end loop over cludters
  }        //end loop over digis
}  //end processlook function

void RecHitProcessor::process(const edm::Event &iEvent,
                              const edm::EventSetup &iSetup,
                              const edm::EDGetToken &recHitToken,
                              const edm::EDGetToken &rpcDigiToken,
                              const edm::EDGetToken &rpcDigiSimLinkToken,
                              const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> &rpcGeomToken,
                              l1t::CPPFDigiCollection &cppfDigis) const {
  // Get the RPC Geometry
  const auto &rpcGeom = iSetup.getData(rpcGeomToken);

  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByToken(rpcDigiToken, rpcDigis);

  // Get the RecHits from the event
  edm::Handle<RPCRecHitCollection> recHits;
  iEvent.getByToken(recHitToken, recHits);

  for (const auto &&rpcdgIt : (*rpcDigis)) {
    const RPCDetId &rpcId = rpcdgIt.first;
    const RPCDigiCollection::Range &range = rpcdgIt.second;
    if (rpcId.region() == 0)
      continue;

    CPPFClusterizer clizer;
    CPPFClusterContainer tcls = clizer.doAction(range);
    CPPFMaskReClusterizer mrclizer;
    CPPFRollMask mask;
    CPPFClusterContainer cls = mrclizer.doAction(rpcId, tcls, mask);

    for (const auto &cl : cls) {
      int region = rpcId.region();
      int isValid = rpcDigis.isValid();
      //      int rawId = rpcId.rawId();
      int Bx = cl.bx();
      const int firststrip = cl.firstStrip();
      const int clustersize = cl.clusterSize();
      const int laststrip = cl.lastStrip();
      const RPCRoll *roll = rpcGeom.roll(rpcId);
      // Get Average Strip position
      const float fstrip = (roll->centreOfStrip(firststrip)).x();
      const float lstrip = (roll->centreOfStrip(laststrip)).x();
      const float centreOfCluster = (fstrip + lstrip) / 2;
      const double y = cl.hasY() ? cl.y() : 0;
      LocalPoint lPos(centreOfCluster, y, 0);
      if (roll->id().region() != 0) {
        const auto &topo = dynamic_cast<const TrapezoidalStripTopology &>(roll->topology());
        const double angle = topo.stripAngle((firststrip + laststrip) / 2.);
        const double x = centreOfCluster - y * std::tan(angle);
        lPos = LocalPoint(x, y, 0);
      }
      const BoundPlane &rollSurface = roll->surface();
      GlobalPoint gPos = rollSurface.toGlobal(lPos);
      float global_theta = emtf::rad_to_deg(gPos.theta().value());
      float global_phi = emtf::rad_to_deg(gPos.phi().value());

      // Endcap region only
      if (region != 0) {
        int int_theta =
            (region == -1 ? 180. * 32. / 36.5 : 0.) + (float)region * global_theta * 32. / 36.5 - 8.5 * 32 / 36.5;
        if (region == 1) {
          if (global_theta < 8.5)
            int_theta = 0;
          if (global_theta > 45.)
            int_theta = 31;
        } else if (region == -1) {
          if (global_theta < 135.)
            int_theta = 31;
          if (global_theta > 171.5)
            int_theta = 0;
        }
        // Local EMTF
        double local_phi = 0.;
        int EMTFsector1 = 0;
        int EMTFsector2 = 0;

        // sector 1
        if ((global_phi > 15.) && (global_phi <= 16.3)) {
          local_phi = global_phi - 15.;
          EMTFsector1 = 1;
          EMTFsector2 = 6;
        } else if ((global_phi > 16.3) && (global_phi <= 53.)) {
          local_phi = global_phi - 15.;
          EMTFsector1 = 1;
          EMTFsector2 = 0;
        } else if ((global_phi > 53.) && (global_phi <= 75.)) {
          local_phi = global_phi - 15.;
          EMTFsector1 = 1;
          EMTFsector2 = 2;
        }
        // sector 2
        else if ((global_phi > 75.) && (global_phi <= 76.3)) {
          local_phi = global_phi - 15.;
          EMTFsector1 = 1;
          EMTFsector2 = 2;
        } else if ((global_phi > 76.3) && (global_phi <= 113.)) {
          local_phi = global_phi - 75.;
          EMTFsector1 = 2;
          EMTFsector2 = 0;
        } else if ((global_phi > 113.) && (global_phi <= 135.)) {
          local_phi = global_phi - 75.;
          EMTFsector1 = 2;
          EMTFsector2 = 3;
        }
        // sector 3
        // less than 180
        else if ((global_phi > 135.) && (global_phi <= 136.3)) {
          local_phi = global_phi - 75.;
          EMTFsector1 = 2;
          EMTFsector2 = 3;
        } else if ((global_phi > 136.3) && (global_phi <= 173.)) {
          local_phi = global_phi - 135.;
          EMTFsector1 = 3;
          EMTFsector2 = 0;
        } else if ((global_phi > 173.) && (global_phi <= 180.)) {
          local_phi = global_phi - 135.;
          EMTFsector1 = 3;
          EMTFsector2 = 4;
        }
        // Greater than -180
        else if ((global_phi < -165.) && (global_phi >= -180.)) {
          local_phi = global_phi + 225.;
          EMTFsector1 = 3;
          EMTFsector2 = 4;
        }
        // Fourth sector
        else if ((global_phi > -165.) && (global_phi <= -163.7)) {
          local_phi = global_phi + 225.;
          EMTFsector1 = 3;
          EMTFsector2 = 4;
        } else if ((global_phi > -163.7) && (global_phi <= -127.)) {
          local_phi = global_phi + 165.;
          EMTFsector1 = 4;
          EMTFsector2 = 0;
        } else if ((global_phi > -127.) && (global_phi <= -105.)) {
          local_phi = global_phi + 165.;
          EMTFsector1 = 4;
          EMTFsector2 = 5;
        }
        // fifth sector
        else if ((global_phi > -105.) && (global_phi <= -103.7)) {
          local_phi = global_phi + 165.;
          EMTFsector1 = 4;
          EMTFsector2 = 5;
        } else if ((global_phi > -103.7) && (global_phi <= -67.)) {
          local_phi = global_phi + 105.;
          EMTFsector1 = 5;
          EMTFsector2 = 0;
        } else if ((global_phi > -67.) && (global_phi <= -45.)) {
          local_phi = global_phi + 105.;
          EMTFsector1 = 5;
          EMTFsector2 = 6;
        }
        // sixth sector
        else if ((global_phi > -45.) && (global_phi <= -43.7)) {
          local_phi = global_phi + 105.;
          EMTFsector1 = 5;
          EMTFsector2 = 6;
        } else if ((global_phi > -43.7) && (global_phi <= -7.)) {
          local_phi = global_phi + 45.;
          EMTFsector1 = 6;
          EMTFsector2 = 0;
        } else if ((global_phi > -7.) && (global_phi <= 15.)) {
          local_phi = global_phi + 45.;
          EMTFsector1 = 6;
          EMTFsector2 = 1;
        }

        int int_phi = int((local_phi + 22.0) * 15. + .5);
        double EMTFLink1 = 0.;
        double EMTFLink2 = 0.;
        double lb = 0.;
        double halfchannel = 0.;

        // Invalid hit
        if (isValid == 0)
          int_phi = 2047;
        // Right integers range
        assert(0 <= int_phi && int_phi < 1250);
        assert(0 <= int_theta && int_theta < 32);

        std::shared_ptr<l1t::CPPFDigi> MainVariables1(new l1t::CPPFDigi(rpcId,
                                                                        Bx,
                                                                        int_phi,
                                                                        int_theta,
                                                                        isValid,
                                                                        lb,
                                                                        halfchannel,
                                                                        EMTFsector1,
                                                                        EMTFLink1,
                                                                        firststrip,
                                                                        clustersize,
                                                                        global_phi,
                                                                        global_theta));
        std::shared_ptr<l1t::CPPFDigi> MainVariables2(new l1t::CPPFDigi(rpcId,
                                                                        Bx,
                                                                        int_phi,
                                                                        int_theta,
                                                                        isValid,
                                                                        lb,
                                                                        halfchannel,
                                                                        EMTFsector2,
                                                                        EMTFLink2,
                                                                        firststrip,
                                                                        clustersize,
                                                                        global_phi,
                                                                        global_theta));
        if (int_theta == 31)
          continue;
        if ((EMTFsector1 > 0) && (EMTFsector2 == 0)) {
          cppfDigis.push_back(*MainVariables1.get());
        }
        if ((EMTFsector1 > 0) && (EMTFsector2 > 0)) {
          cppfDigis.push_back(*MainVariables1.get());
          cppfDigis.push_back(*MainVariables2.get());
        }
        if ((EMTFsector1 == 0) && (EMTFsector2 == 0)) {
          continue;
        }
      }  // No barrel hits
    }    //end loop over clusters
  }      //end loop over digis
}  // End function: void RecHitProcessor::process()
