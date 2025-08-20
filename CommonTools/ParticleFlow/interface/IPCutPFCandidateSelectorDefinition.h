#ifndef CommonTools_ParticleFlow_IPCutPFCandidateSelectorDefinition
#define CommonTools_ParticleFlow_IPCutPFCandidateSelectorDefinition

/**
   \class    pf2pat::IPCutPFCandidateSelectorDefinition IPCutPFCandidateSelectorDefinition.h "CommonTools/ParticleFlow/interface/IPCutPFCandidateSelectorDefinition.h"
   \brief    Selects PFCandidates basing on their compatibility with vertex

   \author   Giovanni Petrucciani
   \version  $Id: IPCutPFCandidateSelectorDefinition.h,v 1.2 2011/04/06 12:12:38 rwolf Exp $
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateSelectorDefinition.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

namespace pf2pat {

  struct IPCutPFCandidateSelectorDefinition : public PFCandidateSelectorDefinition {
    IPCutPFCandidateSelectorDefinition(const edm::ParameterSet &cfg, edm::ConsumesCollector &&iC)
        : verticesToken_(iC.consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"))),
          d0Cut_(cfg.getParameter<double>("d0Cut")),
          dzCut_(cfg.getParameter<double>("dzCut")),
          dtCut_(cfg.getParameter<double>("dtCut")),
          d0SigCut_(cfg.getParameter<double>("d0SigCut")),
          dzSigCut_(cfg.getParameter<double>("dzSigCut")),
          dtSigCut_(cfg.getParameter<double>("dtSigCut")) {}

    static void fillPSetDescription(edm::ParameterSetDescription &desc) {
      desc.add<edm::InputTag>("vertices", edm::InputTag(""));
      desc.add<double>("d0Cut", 0.2)->setComment("transverse IP");
      desc.add<double>("dzCut", 0.5)->setComment("longitudingal IP");
      desc.add<double>("dtCut", -1.0)->setComment("time");
      desc.add<double>("d0SigCut", 99.)->setComment("transverse IP significance");
      desc.add<double>("dzSigCut", 99.)->setComment("longitudingal IP significance");
      desc.add<double>("dtSigCut", -1.0)->setComment("time significance");
    }

    void select(const HandleToCollection &hc, const edm::Event &e, const edm::EventSetup &s) {
      selected_.clear();

      edm::Handle<reco::VertexCollection> vertices;
      e.getByToken(verticesToken_, vertices);
      if (vertices->empty())
        return;
      const reco::Vertex &vtx = (*vertices)[0];
      double vt = vtx.t();
      double vte = vtx.tError();

      unsigned key = 0;
      for (collection::const_iterator pfc = hc->begin(); pfc != hc->end(); ++pfc, ++key) {
        bool passing = true;
        const reco::Track *tk = nullptr;
        if (pfc->gsfTrackRef().isNonnull())
          tk = pfc->gsfTrackRef().get();
        else if (pfc->trackRef().isNonnull())
          tk = pfc->trackRef().get();

        if (tk != nullptr) {
          double d0 = fabs(tk->dxy(vtx.position()));
          double dz = fabs(tk->dz(vtx.position()));
          double d0e = hypot(tk->dxyError(), hypot(vtx.xError(), vtx.yError()));
          double dze = hypot(tk->dzError(), vtx.zError());
          if (d0Cut_ > 0 && d0 > d0Cut_)
            passing = false;
          if (dzCut_ > 0 && dz > dzCut_)
            passing = false;
          if (d0SigCut_ > 0 && d0e > 0 && d0 / d0e > d0SigCut_)
            passing = false;
          if (dzSigCut_ > 0 && dze > 0 && dz / dze > dzSigCut_)
            passing = false;
        }
        double pfct = pfc->time();
        double pfcte = pfc->timeError();
        double dt = fabs(pfct - vt);
        double dte = std::sqrt(pfcte * pfcte + vte * vte);
        if (dtCut_ > 0 && pfcte > 0 && vte > 0 && dt > dtCut_)
          passing = false;
        if (dtSigCut_ > 0 && pfcte > 0 && vte > 0 && dt / dte > dtSigCut_)
          passing = false;

        if (passing) {
          selected_.push_back(reco::PFCandidate(*pfc));
          reco::PFCandidatePtr ptrToMother(hc, key);

          if (pfc->numberOfSourceCandidatePtrs() > 0) {
            selected_.back().setSourceCandidatePtr(edm::Ptr<reco::PFCandidate>(pfc->sourceCandidatePtr(0)));
          } else {
            selected_.back().setSourceCandidatePtr(ptrToMother);
          }
        }
      }
    }

  private:
    edm::EDGetTokenT<reco::VertexCollection> verticesToken_;
    double d0Cut_;
    double dzCut_;
    double dtCut_;
    double d0SigCut_;
    double dzSigCut_;
    double dtSigCut_;
  };
}  // namespace pf2pat

#endif
