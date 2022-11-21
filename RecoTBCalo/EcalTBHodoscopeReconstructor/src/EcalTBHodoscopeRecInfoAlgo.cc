#include "RecoTBCalo/EcalTBHodoscopeReconstructor/interface/EcalTBHodoscopeRecInfoAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <list>

EcalTBHodoscopeRecInfoAlgo::EcalTBHodoscopeRecInfoAlgo(int fitMethod,
                                                       const std::vector<double>& planeShift,
                                                       const std::vector<double>& zPosition)
    : fitMethod_(fitMethod), planeShift_(planeShift), zPosition_(zPosition), myGeometry_() {}

void EcalTBHodoscopeRecInfoAlgo::clusterPos(
    float& x, float& xQuality, const int& ipl, const int& xclus, const int& width) const {
  if (width == 1) {
    // Single fiber
    x = (myGeometry_.getFibreLp(ipl, xclus) + myGeometry_.getFibreRp(ipl, xclus)) / 2.0 - planeShift_[ipl];
    xQuality = (myGeometry_.getFibreRp(ipl, xclus) - myGeometry_.getFibreLp(ipl, xclus));
  } else if (width == 2) {
    // Two half overlapped fibers
    x = (myGeometry_.getFibreLp(ipl, xclus + 1) + myGeometry_.getFibreRp(ipl, xclus)) / 2.0 - planeShift_[ipl];
    xQuality = (myGeometry_.getFibreRp(ipl, xclus) - myGeometry_.getFibreLp(ipl, xclus + 1));
  } else {
    // More then two fibers case
    x = (myGeometry_.getFibreLp(ipl, xclus) + myGeometry_.getFibreRp(ipl, xclus + width - 1)) / 2.0 - planeShift_[ipl];
    xQuality = (myGeometry_.getFibreRp(ipl, xclus + width - 1) - myGeometry_.getFibreLp(ipl, xclus));
  }
}

void EcalTBHodoscopeRecInfoAlgo::fitHodo(float& x,
                                         float& xQuality,
                                         const int& ipl,
                                         const int& nclus,
                                         const std::vector<int>& xclus,
                                         const std::vector<int>& wclus) const {
  if (nclus == 1) {
    // Fill real x as mean position inside the cluster
    // Quality - width of cluster
    // To calculate sigma one can do sigma=sqrt(Quality**2/12.0)
    clusterPos(x, xQuality, ipl, xclus[0], wclus[0]);
  } else {
    xQuality = -10 - nclus;
  }
}

void EcalTBHodoscopeRecInfoAlgo::fitLine(float& x,
                                         float& xSlope,
                                         float& xQuality,
                                         const int& ipl1,
                                         const int& nclus1,
                                         const std::vector<int>& xclus1,
                                         const std::vector<int>& wclus1,
                                         const int& ipl2,
                                         const int& nclus2,
                                         const std::vector<int>& xclus2,
                                         const std::vector<int>& wclus2) const {
  if (nclus1 == 0) {  // Fit with one plane
    fitHodo(x, xQuality, ipl2, nclus2, xclus2, wclus2);
    xSlope = 0.0;  //?? Should we put another number indicating that is not fitted
    return;
  }
  if (nclus2 == 0) {  // Fit with one plane
    fitHodo(x, xQuality, ipl1, nclus1, xclus1, wclus1);
    xSlope = 0.0;  //?? Should we put another number indicating that is not fitted
    return;
  }

  // We have clusters in both planes

  float x1, x2, xQ1, xQ2;
  float xs, x0, xq;

  std::list<BeamTrack> tracks;

  for (int i1 = 0; i1 < nclus1; i1++) {
    for (int i2 = 0; i2 < nclus2; i2++) {
      clusterPos(x1, xQ1, ipl1, xclus1[i1], wclus1[i1]);
      clusterPos(x2, xQ2, ipl2, xclus2[i2], wclus2[i2]);

      xs = (x2 - x1) / (zPosition_[ipl2] - zPosition_[ipl1]);               // slope
      x0 = ((x2 + x1) - xs * (zPosition_[ipl2] + zPosition_[ipl1])) / 2.0;  // x0
      xq = (xQ1 + xQ2) / 2.0;                                               // Quality, how i can do better ?
      tracks.push_back(BeamTrack(x0, xs, xq));
    }
  }

  // find track with minimal slope
  tracks.sort();

  // Return results
  x = tracks.begin()->x;
  xSlope = tracks.begin()->xS;
  xQuality = tracks.begin()->xQ;
}

EcalTBHodoscopeRecInfo EcalTBHodoscopeRecInfoAlgo::reconstruct(const EcalTBHodoscopeRawInfo& hodoscopeRawInfo) const {
  // Reset Hodo data
  float x, y = -100.0;
  float xSlope, ySlope = 0.0;
  float xQuality, yQuality = -100.0;

  int nclus[4];
  std::vector<int> xclus[4];
  std::vector<int> wclus[4];

  for (int ipl = 0; ipl < myGeometry_.getNPlanes(); ipl++) {
    int nhits = hodoscopeRawInfo[ipl].numberOfFiredHits();
    // Finding clusters
    nclus[ipl] = 0;
    if (nhits > 0) {
      int nh = nhits;
      int first = 0;
      int last = 0;
      while (nh > 0) {
        while (hodoscopeRawInfo[ipl][first] == 0)
          first++;  // start
        last = first + 1;
        nh--;
        do {
          while (last < myGeometry_.getNFibres() && hodoscopeRawInfo[ipl][last]) {
            last++;
            nh--;
          }                                                                              //end
          if (last + 1 < myGeometry_.getNFibres() && hodoscopeRawInfo[ipl][last + 1]) {  //Skip 1 fibre hole
            last += 2;
            nh--;
            //std::cout << "Skip fibre " << ipl << " " << first << " "<< last << std::endl;
          } else {
            break;
          }
        } while (nh > 0 && last < myGeometry_.getNFibres());
        wclus[ipl].push_back(last - first);
        xclus[ipl].push_back(first);  // Left edge !!!
        nclus[ipl]++;

        first = last + 1;
      }
    }
    //    printClusters(ipl);
  }

  //! Fit 0
  // Straight line fit for one axis
  if (fitMethod_ == 0) {
    fitLine(x,
            xSlope,
            xQuality,
            0,
            nclus[0],
            xclus[0],
            wclus[0],  // X1
            2,
            nclus[2],
            xclus[2],
            wclus[2]);  // X2
    fitLine(y,
            ySlope,
            yQuality,
            1,
            nclus[1],
            xclus[1],
            wclus[1],  // Y1
            3,
            nclus[3],
            xclus[3],
            wclus[3]);  // Y2
  } else if (fitMethod_ == 1) {
    //! Fit 1
    // x1 and y2 hodoscope
    fitHodo(x, xQuality, 0, nclus[0], xclus[0], wclus[0]);  // X1
    //   if ( xQuality[1] < 0.0 ) {
    //     printFibres(0);
    //     printClusters(0);
    //   }
    fitHodo(y, yQuality, 1, nclus[1], xclus[1], wclus[1]);  // Y1
    //   if ( yQuality[1] < 0.0 ) {
    //     printFibres(1);
    //     printClusters(1);
    //   }
  } else if (fitMethod_ == 2) {
    //! Fit 2
    //x2 and y2 hodoscope
    fitHodo(x, xQuality, 2, nclus[2], xclus[2], wclus[2]);  // X2
    //   if ( xQuality[2] < 0.0 ) {
    //     printFibres(2);
    //     printClusters(2);
    //   }
    fitHodo(y, yQuality, 3, nclus[3], xclus[3], wclus[3]);  // Y2
    //   if ( yQuality[2] < 0.0 ) {
    //     printFibres(3);
    //     printClusters(3);
    //   }
  }

  return EcalTBHodoscopeRecInfo(x, y, xSlope, ySlope, xQuality, yQuality);
}
