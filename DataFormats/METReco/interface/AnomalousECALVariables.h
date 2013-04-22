#ifndef ANOMALOUSECALVARIABLES_H_
#define ANOMALOUSECALVARIABLES_H_
//DataFormats/AnomalousEcalDataFormats/interface/AnomalousECALVariables.h
// system include files
#include <memory>

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/METReco/interface/BoundaryInformation.h"

//using namespace edm;
//using namespace std;
/*
 * This class summarizes the information about the boundary energy calculated in EcalAnomalousEventFilter:
 * 1. next to ECAL border/gap
 * 2. next to masked ECAL channels: for each dead area with boundary energy above a threshold defined in the filter
 * the vector 'v_enDeadNeighbours_EB' or 'v_enDeadNeighbours_EE' is filled with the calculated boundary energy.
 * The determined size of the corresponding cluster is filled in v_enDeadNeighboursNoCells_EB/EE accordingly.
 *
 */
class AnomalousECALVariables {

   public:
      AnomalousECALVariables() {
         //energy next to ECAL Gap
         v_enNeighboursGap_EB.reserve(50);
         v_enNeighboursGap_EE.reserve(50);
         v_enNeighboursGap_EB.clear();
         v_enNeighboursGap_EE.clear();

         //energy around dead cells
         v_boundaryInfoDeadCells_EB = std::vector<BoundaryInformation> ();
         v_boundaryInfoDeadCells_EE = std::vector<BoundaryInformation> ();
         v_boundaryInfoDeadCells_EB.reserve(50);
         v_boundaryInfoDeadCells_EE.reserve(50);
         v_boundaryInfoDeadCells_EB.clear();
         v_boundaryInfoDeadCells_EE.clear();

      }
      ;

      AnomalousECALVariables(const std::vector<BoundaryInformation>& p_enNeighboursGap_EB,
            const std::vector<BoundaryInformation>& p_enNeighboursGap_EE, const std::vector<BoundaryInformation>& p_boundaryInfoDeadCells_EB,
            const std::vector<BoundaryInformation>& p_boundaryInfoDeadCells_EE) {

         v_boundaryInfoDeadCells_EB = std::vector<BoundaryInformation> ();
         v_boundaryInfoDeadCells_EE = std::vector<BoundaryInformation> ();
         v_boundaryInfoDeadCells_EB.reserve(50);
         v_boundaryInfoDeadCells_EE.reserve(50);
         v_boundaryInfoDeadCells_EB.clear();
         v_boundaryInfoDeadCells_EE.clear();
         v_boundaryInfoDeadCells_EB = p_boundaryInfoDeadCells_EB;
         v_boundaryInfoDeadCells_EE = p_boundaryInfoDeadCells_EE;

         v_enNeighboursGap_EB = p_enNeighboursGap_EB;
         v_enNeighboursGap_EE = p_enNeighboursGap_EE;
      }
      ;

      ~AnomalousECALVariables() {
         //cout<<"destructor AnomalousECAL"<<endl;
         v_enNeighboursGap_EB.clear();
         v_enNeighboursGap_EE.clear();
         v_boundaryInfoDeadCells_EB.clear();
         v_boundaryInfoDeadCells_EE.clear();
      }
      ;

      //returns true if at least 1 dead cell area was found in EcalAnomalousEventFilter with
      //boundary energy above threshold
      //Note: no sense to change this cut BELOW the threshold given in EcalAnomalousEventFilter

      bool isDeadEcalCluster(double maxBoundaryEnergy = 10,
            const std::vector<int>& limitDeadCellToChannelStatusEB = std::vector<int> (), const std::vector<int>& limitDeadCellToChannelStatusEE =
                  std::vector<int> ()) const {

         float highestEnergyDepositAroundDeadCell = 0;

         for (int i = 0; i < (int) v_boundaryInfoDeadCells_EB.size(); ++i) {
            BoundaryInformation bInfo = v_boundaryInfoDeadCells_EB[i];

            //check if channel limitation rejectsbInfo
            bool passChannelLimitation = false;
            std::vector<int> status = bInfo.channelStatus;

            for (int cs = 0; cs < (int) limitDeadCellToChannelStatusEB.size(); ++cs) {
               int channelAllowed = limitDeadCellToChannelStatusEB[cs];

               for (std::vector<int>::iterator st_it = status.begin(); st_it != status.end(); ++st_it) {

                  if (channelAllowed == *st_it || (channelAllowed < 0 && abs(channelAllowed) <= *st_it)) {
                     passChannelLimitation = true;
                     break;
                  }
               }
            }

            if (passChannelLimitation || limitDeadCellToChannelStatusEB.size() == 0) {

               if (bInfo.boundaryEnergy > highestEnergyDepositAroundDeadCell) {
                  highestEnergyDepositAroundDeadCell = bInfo.boundaryET;
                  //highestEnergyDepositAroundDeadCell = bInfo.boundaryEnergy;
               }
            }
         }

         for (int i = 0; i < (int) v_boundaryInfoDeadCells_EE.size(); ++i) {
            BoundaryInformation bInfo = v_boundaryInfoDeadCells_EE[i];

            //check if channel limitation rejectsbInfo
            bool passChannelLimitation = false;
            std::vector<int> status = bInfo.channelStatus;

            for (int cs = 0; cs < (int) limitDeadCellToChannelStatusEE.size(); ++cs) {
               int channelAllowed = limitDeadCellToChannelStatusEE[cs];

               for (std::vector<int>::iterator st_it = status.begin(); st_it != status.end(); ++st_it) {

                  if (channelAllowed == *st_it || (channelAllowed < 0 && abs(channelAllowed) <= *st_it)) {
                     passChannelLimitation = true;
                     break;
                  }
               }
            }

            if (passChannelLimitation || limitDeadCellToChannelStatusEE.size() == 0) {

               if (bInfo.boundaryEnergy > highestEnergyDepositAroundDeadCell){
                  highestEnergyDepositAroundDeadCell = bInfo.boundaryET;
                  //highestEnergyDepositAroundDeadCell = bInfo.boundaryEnergy;
               }
            }
         }

         if (highestEnergyDepositAroundDeadCell > maxBoundaryEnergy) {
            //            cout << "<<<<<<<<<< List of EB  Boundary objects <<<<<<<<<<" << endl;
            //            for (int i = 0; i < (int) v_boundaryInfoDeadCells_EB.size(); ++i) {
            //               BoundaryInformation bInfo = v_boundaryInfoDeadCells_EB[i];
            //               cout << "no of neighbouring RecHits:" << bInfo.recHits.size() << endl;
            //               cout << "no of neighbouring DetIds:" << bInfo.detIds.size() << endl;
            //               cout << "boundary energy:" << bInfo.boundaryEnergy << endl;
            //               cout << "Channel stati: ";
            //               for (std::vector<int>::iterator it = bInfo.channelStatus.begin(); it != bInfo.channelStatus.end(); ++it) {
            //                  cout << *it << " ";
            //               }
            //               cout << endl;
            //            }
            //            cout << "<<<<<<<<<< List of EE  Boundary objects <<<<<<<<<<" << endl;
            //            for (int i = 0; i < (int) v_boundaryInfoDeadCells_EE.size(); ++i) {
            //               BoundaryInformation bInfo = v_boundaryInfoDeadCells_EE[i];
            //               cout << "no of neighbouring RecHits:" << bInfo.recHits.size() << endl;
            //               cout << "no of neighbouring DetIds:" << bInfo.detIds.size() << endl;
            //               cout << "boundary energy:" << bInfo.boundaryEnergy << endl;
            //               cout << "Channel stati: ";
            //               for (std::vector<int>::iterator it = bInfo.channelStatus.begin(); it != bInfo.channelStatus.end(); ++it) {
            //                  cout << *it << " ";
            //               }
            //               cout << endl;
            //            }
            return true;
         } else
            return false;
      }

      bool isGapEcalCluster(double maxGapEnergyEB = 10, double maxGapEnergyEE = 10) const {

         float highestEnergyDepositAlongGapEB = 0;

         for (int i = 0; i < (int) v_enNeighboursGap_EB.size(); ++i) {
            BoundaryInformation gapInfo = v_enNeighboursGap_EB[i];

            if (gapInfo.boundaryEnergy > highestEnergyDepositAlongGapEB){
               highestEnergyDepositAlongGapEB = gapInfo.boundaryET;
               //highestEnergyDepositAlongGapEB = gapInfo.boundaryEnergy;
            }
         }

         float highestEnergyDepositAlongGapEE = 0;

         for (int i = 0; i < (int) v_enNeighboursGap_EE.size(); ++i) {
            BoundaryInformation gapInfo = v_enNeighboursGap_EE[i];

            if (gapInfo.boundaryEnergy > highestEnergyDepositAlongGapEE){
               highestEnergyDepositAlongGapEE = gapInfo.boundaryET;
               //highestEnergyDepositAlongGapEE = gapInfo.boundaryEnergy;
            }
         }

         if (highestEnergyDepositAlongGapEB > maxGapEnergyEB || highestEnergyDepositAlongGapEE > maxGapEnergyEE) {
            //            cout << "<<<<<<<<<< List of EB Gap objects <<<<<<<<<<" << endl;
            //            for (int i = 0; i < (int) v_enNeighboursGap_EB.size(); ++i) {
            //               BoundaryInformation gapInfo = v_enNeighboursGap_EB[i];
            //               cout << "no of neighbouring RecHits:" << gapInfo.recHits.size() << endl;
            //               cout << "no of neighbouring DetIds:" << gapInfo.detIds.size() << endl;
            //               cout << "gap energy:" << gapInfo.boundaryEnergy << endl;
            //            }
            //            cout << "<<<<<<<<<< List of EE Gap objects <<<<<<<<<<" << endl;
            //            for (int i = 0; i < (int) v_enNeighboursGap_EE.size(); ++i) {
            //               BoundaryInformation gapInfo = v_enNeighboursGap_EE[i];
            //               cout << "no of neighbouring RecHits:" << gapInfo.recHits.size() << endl;
            //               cout << "no of neighbouring DetIds:" << gapInfo.detIds.size() << endl;
            //               cout << "gap energy:" << gapInfo.boundaryEnergy << endl;
            //            }
            return true;
         } else
            return false;
      }

      std::vector<BoundaryInformation> v_enNeighboursGap_EB;
      std::vector<BoundaryInformation> v_enNeighboursGap_EE;

      std::vector<BoundaryInformation> v_boundaryInfoDeadCells_EB;
      std::vector<BoundaryInformation> v_boundaryInfoDeadCells_EE;

   private:

};

#endif /*ANOMALOUSECALVARIABLES_H_*/
