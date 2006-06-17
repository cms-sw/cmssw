#ifndef HLTReco_HLTFilterObject_h
#define HLTReco_HLTFilterObject_h

/** \class HLTFilterObject
 *
 *
 *  If HLT cuts of intermediate or final HLT filters are satisfied,
 *  instances of this class hold the combination of reconstructed
 *  physics objects (e/gamma/mu/jet/MMet...) satisfying the cuts.
 *
 *  This implementation is not completely space-efficient as some
 *  physics object containers may stay empty. However, the big
 *  advantage is that the solution is generic, i.e., works for all
 *  possible HLT filters. Hence we accept the reasonably small
 *  overhead of empty containers.
 *
 *  $Date: 2006/06/17 00:18:35 $
 *  $Revision: 1.9 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/HLTReco/interface/HLTParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <cassert>
#include <vector>
#include <map>

namespace reco
{
  using namespace std;

  class HLTFilterObjectBase {

  private:
    unsigned int index_; // packed path (24 bits) / module (8 bits) index

  public:

    HLTFilterObjectBase(): index_() { }
    HLTFilterObjectBase(unsigned int p, unsigned int m): index_(256*p+m) {
      assert(p<256*65536); 
      assert(m<256);
    }

    inline unsigned int path()   const { return index_/256; }
    inline unsigned int module() const { return index_%256; }

  };

  class HLTFilterObject : public HLTFilterObjectBase {

  typedef edm::hlt::HLTScalar HLTScalar;

  private:
    map<HLTScalar,float>  scalars_;   // scalar quantities used in filter (HT, ...)
    vector<HLTParticle>   particles_; // particles/MET (4-momentum vectors) used by filter

  public:

    HLTFilterObject(): HLTFilterObjectBase(), scalars_(), particles_() { }

    unsigned int numberScalars  () const { return   scalars_.size();}
    unsigned int numberParticles() const { return particles_.size();}

    bool getScalar(const HLTScalar scalar, float& value) const {
      if (scalars_.find(scalar)==scalars_.end()) {
        return false;
      } else {
        value = scalars_.find(scalar)->second;
        return true;
      }
    }
    void putScalar(const HLTScalar scalar, const float value) {
      scalars_[scalar] = value;
    }

    const HLTParticle& getParticle(const unsigned int i) const {
      return particles_.at(i);
    }

    void putParticle(const edm::RefToBase<Candidate>& ref) {
      particles_.push_back(HLTParticle(*ref));
    }

  };


  class HLTFilterObjectWithRefs : public HLTFilterObject {

  private:
    std::vector<edm::RefToBase<Candidate> > refs_;

  public:

    HLTFilterObjectWithRefs(): HLTFilterObject(), refs_() { }

    void putParticle(const edm::RefToBase<Candidate>& ref) {
      this->HLTFilterObject::putParticle(ref);
      refs_.push_back(ref);
    }

    const edm::RefToBase<Candidate>& getParticleRef(const unsigned int i) const {
      return refs_.at(i);
    }
 
  };
}

#endif
