// $Id:$

#ifndef QcdLowPtDQM_H
#define QcdLowPtDQM_H

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

class DQMStore;
class MonitorElement;
class TrackerGeometry;

class QcdLowPtDQM : public edm::EDAnalyzer 
{
  public:
    class Pixel {
      public:
        Pixel(double x=0, double y=0, double z=0, double eta=0, double phi=0) : 
          x_(x), y_(y), z_(z), eta_(eta), phi_(phi) {}
        Pixel(const GlobalPoint &p) :
          x_(p.x()), y_(p.y()), z_(p.z()), eta_(p.eta()), phi_(p.phi()) {}
        double x()   const { return x_;   }
        double y()   const { return y_;   }
        double z()   const { return z_;   }
        double eta() const { return eta_; }
        double phi() const { return phi_; }
      protected:    
        double x_,y_,z_,eta_,phi_;
    };
    class Tracklet {
      public:
        Tracklet() : i1_(-1), i2_(-2), deta_(0), dphi_(0) {}
        Tracklet(const Pixel &p1, const Pixel &p2) : 
          p1_(p1), p2_(p2), i1_(-1), i2_(-1), 
          deta_(p1.eta()-p2.eta()), dphi_(Geom::deltaPhi(p1.phi(),p2.phi())) {}
        int          i1()    const { return i1_;   }
        int          i2()    const { return i2_;   }
        double       deta()  const { return deta_; }
        double       dphi()  const { return dphi_; }
        const Pixel &p1()    const { return p1_;   }
        const Pixel &p2()    const { return p2_;   }
        void         seti1(int i1) { i1_ = i1;     }      
        void         seti2(int i2) { i2_ = i2;     }      
      protected:
        Pixel  p1_,p2_;
        int    i1_, i2_;
        double deta_,dphi_;
    };
    class Vertex {
      public:
        Vertex(double x=0, double y=0, double z=0, double xs=0, double ys=0, double zs=0, int n=0) :
          x_(x), y_(y), z_(z), xs_(xs), ys_(ys), zs_(zs), n_(n) {}
        int    n()   const { return n_;  }
        double x()   const { return x_;  }
        double y()   const { return y_;  }
        double z()   const { return z_;  }
        double xs()  const { return xs_; }
        double ys()  const { return ys_; }
        double zs()  const { return zs_; }
        void   set(int n, double z, double zs) { n_=n; z_ =z; zs_ = zs; }
        void   set(int n, double x,double y,double z, double xs,double ys,double zs) 
                 { n_=n; x_ =x; xs_ = xs; y_ =y; ys_ = ys; z_ =z; zs_ = zs; }
      protected:    
        double x_,y_,z_,xs_,ys_,zs_;
        int n_;
    };

    QcdLowPtDQM(const edm::ParameterSet&);
    virtual ~QcdLowPtDQM();

    void                        beginJob(void);
    void                        beginRun(const edm::Run &r, const edm::EventSetup &iSetup);
    void                        analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup);
    void                        endJob(void);
    void                        endRun(const edm::Run &r, const edm::EventSetup &iSetup);

  private:
    void                        bookHistos();
    void                        fillHltBits(const edm::Event &iEvent);
    void                        fillPixels(const edm::Event &iEvent);
    void                        fillTracklets(const edm::Event &iEvent, int which=12);
    void                        fillTracklets(std::vector<Tracklet> &tracklets, 
                                              std::vector<Pixel> &pix1, std::vector<Pixel> &pix2);
    template <typename TYPE>
    void                        getProduct(const std::string name, edm::Handle<TYPE> &prod,
                                           const edm::Event &event) const;    
    template <typename TYPE>
    bool                        getProductSafe(const std::string name, edm::Handle<TYPE> &prod,
                                               const edm::Event &event) const;
    void                        print(int level, const char *msg);
    void                        print(int level, const std::string &msg)
                                  { print(level,msg.c_str()); }
    void                        reallyPrint(int level, const char *msg);
    void                        trackletVertexUnbinned();

    std::string                 hltResName_;    //HLT trigger results name
    std::string                 hltProcName_;   //HLT process name
    std::vector<std::string>    hltTrgNames_;   //HLT trigger name(s)
    std::string                 pixelName_;     //pixel reconstructed hits name
    double                      ZVCut_;         //Z vertex cut for selected events
    double                      dPhiVc_;        //dPhi vertex cut for tracklet based vertex
    double                      dZVc_;          //dZ vertex cut for tracklet based vertex
    int                         verbose_;       //verbosity (0=debug,1=warn,2=error,3=exceptions)
    bool                        usePixelQ_;      //if true use pixel quality word

    std::vector<int>            hltTrgBits_;    //HLT trigger bit(s)
    std::vector<bool>           hltTrgDeci_;    //HLT trigger descision(s)
    std::vector<Pixel>          bpix1_;         //barrel pixels layer 1
    std::vector<Pixel>          bpix2_;         //barrel pixels layer 2
    std::vector<Pixel>          bpix3_;         //barrel pixels layer 3
    std::vector<Tracklet>       btracklets12_;  //barrel tracklets 12
    std::vector<Tracklet>       btracklets13_;  //barrel tracklets 13
    std::vector<Tracklet>       btracklets23_;  //barrel tracklets 23
    Vertex                      trackletV_;     //reconstructed tracklet vertex

    const TrackerGeometry      *tgeo_;             //tracker geometry
    DQMStore                   *theDbe_;           //dqm store
    MonitorElement             *hNhitsL1_;         //number of hits on layer 1
    MonitorElement             *hNhitsL2_;         //number of hits on layer 2
    MonitorElement             *hNhitsL3_;         //number of hits on layer 3
    MonitorElement             *hdNdEtaHitsL1_;    //dN/dEta of hits on layer 1
    MonitorElement             *hdNdEtaHitsL2_;    //dN/dEta of hits on layer 2
    MonitorElement             *hdNdEtaHitsL3_;    //dN/dEta of hits on layer 3
    MonitorElement             *hdNdPhiHitsL1_;    //dN/dPhi of hits on layer 1
    MonitorElement             *hdNdPhiHitsL2_;    //dN/dPhi of hits on layer 2
    MonitorElement             *hdNdPhiHitsL3_;    //dN/dPhi of hits on layer 3
    MonitorElement             *hTrkVtxZ_;         //tracklet z vertex
    MonitorElement             *hTrkRawdEtadPhi;   //tracklet dEta/dPhi distribution
    MonitorElement             *hTrkRawdEta;       //tracklet dEta distribution
    MonitorElement             *hTrkRawddPhi;      //tracklet dPhi distribution
    MonitorElement             *h2TrigCorr_;       //trigger correlation plot 
};

//--------------------------------------------------------------------------------------------------
template <typename TYPE>
inline void QcdLowPtDQM::getProduct(const std::string name, edm::Handle<TYPE> &prod,
                                    const edm::Event &event) const
{
  // Try to access data collection from EDM file. We check if we really get just one
  // product with the given name. If not we throw an exception.

  event.getByLabel(edm::InputTag(name),prod);
  if (!prod.isValid()) 
    throw edm::Exception(edm::errors::Configuration, "QcdLowPtDQM::GetProduct()\n")
      << "Collection with label " << name << " is not valid" <<  std::endl;
}

//--------------------------------------------------------------------------------------------------
template <typename TYPE>
inline bool QcdLowPtDQM::getProductSafe(const std::string name, edm::Handle<TYPE> &prod,
                                        const edm::Event &event) const
{
  // Try to safely access data collection from EDM file. We check if we really get just one
  // product with the given name. If not, we return false.

  try {
    event.getByLabel(edm::InputTag(name),prod);
    if (!prod.isValid()) 
      return false;
  } catch (...) {
    return false;
  }
  return true;
}

//--------------------------------------------------------------------------------------------------
inline void QcdLowPtDQM::print(int level, const char *msg)
{
  // Print out message if it 

  if (level>=verbose_) 
    reallyPrint(level,msg);
}
#endif
