#ifndef AlCaHOCalibProducer_HOCalibVariables_h
#define AlCaHOCalibProducer_HOCalibVariables_h
#include <vector>
//April 2015 : Added more variables, e.g., momatho, tkpt03, ecal03, hcal03, pileup, nprim
class HOCalibVariables {
public:
  int nmuon;  //number of muons in the event
  int nprim;  // number of primary vertices

  float pileup;  //Number of pileup events in the bunch crossing

  float trkdr;  //r-phi coordinate of track wrt vertex
  float trkdz;  //Z coordinate of track wrt vertex

  float trkvx;  //X-Position of fitted track in the inner layer of DT
  float trkvy;  //Y-Position of fitted track in the inner layer of DT
  float trkvz;  //Z-Position of fitted track in the inner layer of DT

  float momatho;  //Magnitude of track momenta at HO
  float trkmm;    //Magnitude of track momenta
  float trkth;    //Polar angle of track
  float trkph;    //Azimuthal angle of track

  float ndof;  //number of degrees of freedom (track fitting)
  //  float nrecht; //Number of rechit candidates in the track
  float chisq;  //Fitted normalised chisquare (chi^2/ndf)

  float therr;  //Error in fitted polar angle
  float pherr;  //Error in fitted azimuthal angle

  int isect;            //HO tile information 100*abs(ieta+30)+abs(iphi) : Standalong geometry
  int isect2;           //HO tile information 100*abs(ieta+30)+abs(iphi) : CMSSW Geometry
  float hodx;           //Minimum distance of tracks entrace point in tiles local
                        // co-ordinate system from an edge in X-axis
  float hody;           //Same in Y-axis
  float hoang;          //Angle between track (in HO tiles) and HO Z-axis
  float htime;          //Energy weighted time of signal
  unsigned int hoflag;  //(HORecHitCollection::const_iterator jk, (*jk).flags();
  float hosig[9];       //HO signal in 3x3 tower with respect to the tile, where
                        // muon passed through (for the consistency check of
                        // track reconstruction

  float hocorsig[18];  //Signals in all 18 pixel in that HPD, where muon signal is
                       //expected. One is signal, remaings are either cross-talk or
                       //due to wrongly reconstructed tracks (For cross talk study)
                       //For Ring 0 hocorsig[16-17] are X-Y position in layer0
  float hocro;         //Signal in tile with same eta, but phi1=phi+6
                       // (a check of pedestal)
  float hbhesig[9];    //Signal in HB towers
  float caloen[3];     //Associated energy in the calorimeter, 15, 25, 35 degree
  float tkpt03;        //Energy deposit in tracker
  float ecal03;        //Energy deposit in EM
  float hcal03;        //Energy deposit in HCAL
};

typedef std::vector<HOCalibVariables> HOCalibVariableCollection;

#endif
