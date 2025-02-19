#ifndef DataFormats_Candidate_VertexCompositeCandidate_H
#define DataFormats_Candidate_VertexCompositeCandidate_H
/** \class reco::VertexCompositeCandidate
 *
 * A composite Candidate  with error
 * matrix and other vertex fix information. 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: VertexCompositeCandidate.h,v 1.9 2009/11/02 21:46:53 srappocc Exp $
 *
 */
#include "DataFormats/Candidate/interface/VertexCompositeCandidateFwd.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

namespace reco {
  class VertexCompositeCandidate : public CompositeCandidate {
  public:
    VertexCompositeCandidate() : CompositeCandidate() { }
    /// constructor from values
    VertexCompositeCandidate(Charge q, const LorentzVector & p4, const Point & vtx,
			     int pdgId = 0, int status = 0, bool integerCharge = true) :
      CompositeCandidate(q, p4, vtx, pdgId, status, integerCharge),
      chi2_(0), ndof_(0) { }
    /// constructor from values
    VertexCompositeCandidate(Charge q, const LorentzVector & p4, const Point & vtx,
			     const CovarianceMatrix & err, double chi2, double ndof,
			     int pdgId = 0, int status = 0, bool integerCharge = true);
     /// constructor from values
    explicit VertexCompositeCandidate(const Candidate & p) :
      CompositeCandidate(p), chi2_(0), ndof_(0) { }
     /// constructor from values
    explicit VertexCompositeCandidate(const CompositeCandidate & p) :
      CompositeCandidate(p), chi2_(0), ndof_(0) { }
    /// destructor
    virtual ~VertexCompositeCandidate();
    /// returns a clone of the candidate
    virtual VertexCompositeCandidate * clone() const;
    /// chi-squares
    virtual double vertexChi2() const { return chi2_; }
    /** Number of degrees of freedom
     *  Meant to be Double32_t for soft-assignment fitters: 
     *  tracks may contribute to the vertex with fractional weights.
     *  The ndof is then = to the sum of the track weights.
     *  see e.g. CMS NOTE-2006/032, CMS NOTE-2004/002
     */
    virtual double vertexNdof() const { return ndof_; }
    /// chi-squared divided by n.d.o.f.
    virtual double vertexNormalizedChi2() const { return chi2_ / ndof_; }
    /// (i, j)-th element of error matrix, i, j = 0, ... 2
    virtual double vertexCovariance(int i, int j) const { 
      return covariance_[idx(i, j)]; 
    }
    using reco::LeafCandidate::vertexCovariance; // avoid hiding the
    /// fill SMatrix
    virtual void fillVertexCovariance(CovarianceMatrix & v) const;
    /// set chi2 and ndof
    void setChi2AndNdof(double chi2, double ndof) {
      chi2_ = chi2; ndof_ = ndof;
    }
    /// set covariance matrix
    void setCovariance(const CovarianceMatrix &m);
  private:
    /// chi-sqared
    Double32_t chi2_;
    /// number of degrees of freedom
    Double32_t ndof_;
    /// covariance matrix (3x3) as vector
    Double32_t covariance_[size];
    /// position index
    index idx(index i, index j) const {
      int a = (i <= j ? i : j), b = (i <= j ? j : i);
      return b * (b + 1)/2 + a;
    }
  };

}

#endif
