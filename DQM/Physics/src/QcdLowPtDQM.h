// $Id: QcdLowPtDQM.h,v 1.10 2010/03/03 09:32:40 olzem Exp $

#ifndef QcdLowPtDQM_H
#define QcdLowPtDQM_H

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include <TMath.h>
#include <vector>

class DQMStore;
class MonitorElement;
class TrackerGeometry;
class TH1F;
class TH2F;
class TH3F;

class QcdLowPtDQM : public edm::EDAnalyzer 
{
  public:
    class Pixel {
      public:
        Pixel(double x=0, double y=0, double z=0, double eta=0, double phi=0, 
              double adc=0, double sx=0, double sy=0) : 
          x_(x), y_(y), z_(z), rho_(TMath::Sqrt(x_*x_+y_*y_)), 
          eta_(eta), phi_(phi), adc_(adc), sizex_(sx), sizey_(sy) {}
        Pixel(const GlobalPoint &p, double adc=0, double sx=0, double sy=0) :
          x_(p.x()), y_(p.y()), z_(p.z()), rho_(TMath::Sqrt(x_*x_+y_*y_)), 
          eta_(p.eta()), phi_(p.phi()), adc_(adc), sizex_(sx), sizey_(sy) {}
        double adc()   const { return adc_;   }
        double eta()   const { return eta_;   }
        double rho()   const { return rho_;   }
        double phi()   const { return phi_;   }
        double sizex() const { return sizex_; }
        double sizey() const { return sizey_; }
        double x()     const { return x_;     }
        double y()     const { return y_;     }
        double z()     const { return z_;     }
      protected:    
        double x_,y_,z_,rho_,eta_,phi_;
        double adc_,sizex_,sizey_;
    };
    class Tracklet {
      public:
        Tracklet() : i1_(-1), i2_(-2), deta_(0), dphi_(0) {}
        Tracklet(const Pixel &p1, const Pixel &p2) : 
          p1_(p1), p2_(p2), i1_(-1), i2_(-1), 
          deta_(p1.eta()-p2.eta()), dphi_(Geom::deltaPhi(p1.phi(),p2.phi())) {}
        double       deta()  const { return deta_;     }
        double       dphi()  const { return dphi_;     }
        int          i1()    const { return i1_;       }
        int          i2()    const { return i2_;       }
        double       eta()   const { return p1_.eta(); }
        const Pixel &p1()    const { return p1_;       }
        const Pixel &p2()    const { return p2_;       }
        void         seti1(int i1) { i1_ = i1;         }      
        void         seti2(int i2) { i2_ = i2;         }      
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
        void   set(int n, double z, double zs) { n_= n; z_ = z; zs_ = zs; }
        void   set(int n, double x,double y,double z, double xs,double ys,double zs) 
                 { n_= n; x_ = x; xs_ = xs; y_ = y; ys_ = ys; z_ = z; zs_ = zs; }
      protected:    
        double x_,y_,z_,xs_,ys_,zs_;
        int n_;
    };

    QcdLowPtDQM(const edm::ParameterSet &parameters);
    virtual ~QcdLowPtDQM();

    void                          analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup);
    void                          beginJob(void);
    void                          beginLuminosityBlock(const edm::LuminosityBlock &l, 
                                                       const edm::EventSetup &iSetup);
    void                          beginRun(const edm::Run &r, const edm::EventSetup &iSetup);
    void                          endJob(void);
    void                          endRun(const edm::Run &r, const edm::EventSetup &iSetup);
    void                          endLuminosityBlock(const edm::LuminosityBlock &l, 
                                                     const edm::EventSetup &iSetup);

  private:
    void                          book1D(std::vector<MonitorElement*> &mes, 
                                         const std::string &name, const std::string &title, 
                                         int nx, double x1, double x2, bool sumw2=1, bool sbox=1);
    void                          book2D(std::vector<MonitorElement*> &mes, 
                                         const std::string &name, const std::string &title, 
                                         int nx, double x1, double x2, int ny, double y1, double y2,
                                         bool sumw2=1, bool sbox=1);
    void                          create1D(std::vector<TH1F*> &mes, 
                                           const std::string &name, const std::string &title, 
                                           int nx, double x1, double x2, bool sumw2=1, bool sbox=1);
    void                          create2D(std::vector<TH2F*> &mes, 
                                           const std::string &name, const std::string &title, 
                                           int nx, double x1, double x2, int ny, double y1, double y2,
                                           bool sumw2=1, bool sbox=1);
    void                          createHistos();
    void                          fill1D(std::vector<TH1F*> &hs, double val, double w=1.);
    void                          fill1D(std::vector<MonitorElement*> &mes, double val, double w=1.);
    void                          fill2D(std::vector<TH2F*> &hs, 
                                         double valx, double valy, double w=1.);
    void                          fill2D(std::vector<MonitorElement*> &mes, 
                                         double valx, double valy, double w=1.);
    void                          fill3D(std::vector<TH3F*> &hs, int gbin, double w=1.);
    void                          filldNdeta(const TH3F *AlphaTracklets,
                                             const std::vector<TH3F*> &NsigTracklets,
                                             const std::vector<TH3F*> &NbkgTracklets,
                                             const std::vector<TH1F*> &NEvsPerEta,
                                             std::vector<MonitorElement*> &hdNdEtaRawTrkl,
                                             std::vector<MonitorElement*> &hdNdEtaSubTrkl,
                                             std::vector<MonitorElement*> &hdNdEtaTrklets);
    void                          fillHltBits(const edm::Event &iEvent);
    void                          fillPixels(const edm::Event &iEvent, const edm::EventSetup& iSetup);
    void                          fillPixelClusterInfos(const edm::Event &iEvent, int which=12);
    void                          fillPixelClusterInfos(const double vz,
                                                        const std::vector<Pixel> &pix, 
                                                        std::vector<MonitorElement*> &hClusterYSize,
                                                        std::vector<MonitorElement*> &hClusterADC);
    void                          fillTracklets(const edm::Event &iEvent, int which=12);
    void                          fillTracklets(std::vector<Tracklet> &tracklets, 
                                                const std::vector<Pixel> &pix1, 
                                                const std::vector<Pixel> &pix2,
                                                const Vertex &trackletV);
    void                          fillTracklets(const std::vector<Tracklet> &tracklets, 
                                                const std::vector<Pixel> &pixels,
                                                const Vertex &trackletV,
                                                const TH3F *AlphaTracklets,
                                                std::vector<TH3F*> &NsigTracklets,
                                                std::vector<TH3F*> &NbkgTracklets,
                                                std::vector<TH1F*> &eventpereta,
                                                std::vector<MonitorElement*> &detaphi,
                                                std::vector<MonitorElement*> &deta,
                                                std::vector<MonitorElement*> &dphi,
                                                std::vector<MonitorElement*> &etavsvtx);
    template <typename TYPE>
    void                          getProduct(const std::string name, edm::Handle<TYPE> &prod,
                                             const edm::Event &event) const;    
    template <typename TYPE>
    bool                          getProductSafe(const std::string name, edm::Handle<TYPE> &prod,
                                                 const edm::Event &event) const;
    void                          print(int level, const char *msg);
    void                          print(int level, const std::string &msg)
                                    { print(level,msg.c_str()); }
    void                          reallyPrint(int level, const char *msg);
    void                          trackletVertexUnbinned(const edm::Event &iEvent, int which=12);
    void                          trackletVertexUnbinned(std::vector<Pixel> &pix1, 
                                                         std::vector<Pixel> &pix2,
                                                         Vertex &vtx);
    double                        vertexZFromClusters(const std::vector<Pixel> &pix) const;
    void                          yieldAlphaHistogram(int which=12);

    std::string                   hltResName_;         //HLT trigger results name
    std::vector<std::string>      hltProcNames_;       //HLT process name(s)
    std::vector<std::string>      hltTrgNames_;        //HLT trigger name(s)
    std::string                   pixelName_;          //pixel reconstructed hits name
    std::string                   clusterVtxName_;     //cluster vertex name
    double                        ZVCut_;              //Z vertex cut for selected events
    double                        ZVEtaRegion_;        //Z vertex eta region
    double                        ZVVtxRegion_;        //Z vertex vtx region
    double                        dPhiVc_;             //dPhi vertex cut for tracklet based vertex
    double                        dZVc_;               //dZ vertex cut for tracklet based vertex
    double                        sigEtaCut_;          //signal tracklet eta cut
    double                        sigPhiCut_;          //signal tracklet phi cut
    double                        bkgEtaCut_;          //bkg tracklet eta cut
    double                        bkgPhiCut_;          //bgk tracklet phi cut
    int                           verbose_;            //verbosity (0=debug,1=warn,2=error,3=throw)
    int                           pixLayers_;          //12 for 12, 13 for 12 and 13, 23 for all 
    int                           clusLayers_;         //12 for 12, 13 for 12 and 13, 23 for all 
    bool                          useRecHitQ_;         //if true use rec hit quality word
    bool                          usePixelQ_;          //if true use pixel hit quality word
    std::vector<int>              hltTrgBits_;         //HLT trigger bit(s)
    std::vector<bool>             hltTrgDeci_;         //HLT trigger descision(s)
    std::vector<std::string>      hltTrgUsedNames_;    //HLT used trigger name(s)
    std::string                   hltUsedResName_;     //used HLT trigger results name
    std::vector<Pixel>            bpix1_;              //barrel pixels layer 1
    std::vector<Pixel>            bpix2_;              //barrel pixels layer 2
    std::vector<Pixel>            bpix3_;              //barrel pixels layer 3
    std::vector<Tracklet>         btracklets12_;       //barrel tracklets 12
    std::vector<Tracklet>         btracklets13_;       //barrel tracklets 13
    std::vector<Tracklet>         btracklets23_;       //barrel tracklets 23
    Vertex                        trackletV12_;        //reconstructed tracklet vertex 12
    Vertex                        trackletV13_;        //reconstructed tracklet vertex 13
    Vertex                        trackletV23_;        //reconstructed tracklet vertex 23
    std::vector<TH3F*>            NsigTracklets12_;    //number of signal tracklets 12
    std::vector<TH3F*>            NbkgTracklets12_;    //number of background tracklets 12
    std::vector<TH1F*>            hEvtCountsPerEta12_; //event count per tracklet12 eta-vtx region
    TH3F                         *AlphaTracklets12_;   //alpha correction for tracklets 12
    std::vector<TH3F*>            NsigTracklets13_;    //number of signal tracklets 13
    std::vector<TH3F*>            NbkgTracklets13_;    //number of background tracklets 13
    std::vector<TH1F*>            hEvtCountsPerEta13_; //event count per tracklet13 eta-vtx region 
    TH3F                         *AlphaTracklets13_;   //alpha correction for tracklets 13
    std::vector<TH3F*>            NsigTracklets23_;    //number of signal tracklets 23
    std::vector<TH3F*>            NbkgTracklets23_;    //number of background tracklets 23
    std::vector<TH1F*>            hEvtCountsPerEta23_; //event count per tracklet23 eta-vtx region
    TH3F                         *AlphaTracklets23_;   //alpha correction for tracklets 23
    HLTConfigProvider             hltConfig_;
    const TrackerGeometry        *tgeo_;               //tracker geometry
    DQMStore                     *theDbe_;             //dqm store
    MonitorElement               *repSumMap_;          //report summary map
    MonitorElement               *repSummary_;         //report summary
    MonitorElement               *h2TrigCorr_;         //trigger correlation plot 
    std::vector<MonitorElement*>  hNhitsL1_;           //number of hits on layer 1
    std::vector<MonitorElement*>  hNhitsL2_;           //number of hits on layer 2
    std::vector<MonitorElement*>  hNhitsL3_;           //number of hits on layer 3
    std::vector<MonitorElement*>  hNhitsL1z_;          //number of hits on layer 1 (zoomed)
    std::vector<MonitorElement*>  hNhitsL2z_;          //number of hits on layer 2 (zoomed)
    std::vector<MonitorElement*>  hNhitsL3z_;          //number of hits on layer 3 (zoomed)
    std::vector<MonitorElement*>  hdNdEtaHitsL1_;      //dN/dEta of hits on layer 1
    std::vector<MonitorElement*>  hdNdEtaHitsL2_;      //dN/dEta of hits on layer 2
    std::vector<MonitorElement*>  hdNdEtaHitsL3_;      //dN/dEta of hits on layer 3
    std::vector<MonitorElement*>  hdNdPhiHitsL1_;      //dN/dPhi of hits on layer 1
    std::vector<MonitorElement*>  hdNdPhiHitsL2_;      //dN/dPhi of hits on layer 2
    std::vector<MonitorElement*>  hdNdPhiHitsL3_;      //dN/dPhi of hits on layer 3
    std::vector<MonitorElement*>  hTrkVtxZ12_;         //tracklet z vertex 12 histograms
    std::vector<MonitorElement*>  hTrkVtxZ13_;         //tracklet z vertex 13 histograms
    std::vector<MonitorElement*>  hTrkVtxZ23_;         //tracklet z vertex 23 histograms
    std::vector<MonitorElement*>  hRawTrkEtaVtxZ12_;   //tracklet eta vs z vertex 12 histograms
    std::vector<MonitorElement*>  hRawTrkEtaVtxZ13_;   //tracklet eta vs z vertex 13 histograms
    std::vector<MonitorElement*>  hRawTrkEtaVtxZ23_;   //tracklet eta vs z vertex 23 histograms
    std::vector<MonitorElement*>  hTrkRawDetaDphi12_;  //tracklet12 Deta/Dphi distribution
    std::vector<MonitorElement*>  hTrkRawDeta12_;      //tracklet12 Deta distribution
    std::vector<MonitorElement*>  hTrkRawDphi12_;      //tracklet12 Dphi distribution
    std::vector<MonitorElement*>  hTrkRawDetaDphi13_;  //tracklet13 Deta/Dphi distribution
    std::vector<MonitorElement*>  hTrkRawDeta13_;      //tracklet13 Deta distribution
    std::vector<MonitorElement*>  hTrkRawDphi13_;      //tracklet13 Dphi distribution
    std::vector<MonitorElement*>  hTrkRawDetaDphi23_;  //tracklet23 Deta/Dphi distribution
    std::vector<MonitorElement*>  hTrkRawDeta23_;      //tracklet23 Deta distribution
    std::vector<MonitorElement*>  hTrkRawDphi23_;      //tracklet23 Dphi distribution
    std::vector<MonitorElement*>  hdNdEtaRawTrkl12_;   //dN/dEta from raw tracklets 12 
    std::vector<MonitorElement*>  hdNdEtaSubTrkl12_;   //dN/dEta from beta tracklets 12 
    std::vector<MonitorElement*>  hdNdEtaTrklets12_;   //dN/dEta corrected by alpha 12
    std::vector<MonitorElement*>  hdNdEtaRawTrkl13_;   //dN/dEta from raw tracklets 13
    std::vector<MonitorElement*>  hdNdEtaSubTrkl13_;   //dN/dEta from beta tracklets 13
    std::vector<MonitorElement*>  hdNdEtaTrklets13_;   //dN/dEta corrected by alpha 13
    std::vector<MonitorElement*>  hdNdEtaRawTrkl23_;   //dN/dEta from raw tracklets 23
    std::vector<MonitorElement*>  hdNdEtaSubTrkl23_;   //dN/dEta from beta tracklets 23
    std::vector<MonitorElement*>  hdNdEtaTrklets23_;   //dN/dEta corrected by alpha 23
    std::vector<MonitorElement*>  hClusterVertexZ_;    //cluster z vertex histograms
    std::vector<MonitorElement*>  hClusterYSize1_;     //cluster y size histograms on layer 1
    std::vector<MonitorElement*>  hClusterYSize2_;     //cluster y size histograms on layer 2
    std::vector<MonitorElement*>  hClusterYSize3_;     //cluster y size histograms on layer 3
    std::vector<MonitorElement*>  hClusterADC1_;       //cluster adc histograms on layer 1
    std::vector<MonitorElement*>  hClusterADC2_;       //cluster adc histograms on layer 2
    std::vector<MonitorElement*>  hClusterADC3_;       //cluster adc histograms on layer 3
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

  if (name.size()==0)
    return false;

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
