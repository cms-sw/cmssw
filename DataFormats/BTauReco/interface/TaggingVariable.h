#ifndef DataFormats_BTauReco_TaggingVariable_h
#define DataFormats_BTauReco_TaggingVariable_h

#include <utility>
#include <vector>
#include <string>

#include <boost/static_assert.hpp>
#include <boost/pointee.hpp>
#include <boost/type_traits/is_convertible.hpp>

#include "DataFormats/BTauReco/interface/RefMacros.h"

namespace reco {

  namespace btau {

    // define the enum in a namespace to avoid polluting reco with all the enum values
    enum TaggingVariableName {
      jetEnergy = 0,                            // jet energy
      jetPt,                                    // jet transverse momentum
      trackJetPt,                               // track-based jet transverse momentum
      jetEta,                                   // jet pseudorapidity
      jetPhi,                                   // jet polar angle
      jetNTracks,                               // tracks associated to jet

      trackMomentum,                            // track momentum
      trackEta,                                 // track pseudorapidity
      trackPhi,                                 // track polar angle

      trackPtRel,                               // track transverse momentum, relative to the jet axis
      trackPPar,                                // track parallel momentum, along the jet axis
      trackEtaRel,                              // track pseudorapidity, relative to the jet axis
      trackDeltaR,                              // track pseudoangular distance from the jet axis
      trackPtRatio,                             // track transverse momentum, relative to the jet axis, normalized to its energy
      trackPParRatio,                           // track parallel momentum, along the jet axis, normalized to its energy

      trackIp2dSig,                             // track 2D impact parameter signifncance
      trackSip2dVal,                            // track 2D signed impact parameter
      trackSip2dSig,                            // track 2D signed impact parameter significance
      trackSip3dVal,                            // track 3D signed impact parameter
      trackSip3dSig,                            // track 3D signed impact parameter significance
      trackDecayLenVal,                         // track decay length
      trackDecayLenSig,                         // track decay length significance
      trackJetDistVal,                          // minimum track approach distance to jet axis
      trackJetDistSig,                          // minimum track approach distance to jet axis significance
      trackGhostTrackDistVal,			// minimum approach distance to ghost track
      trackGhostTrackDistSig,			// minimum approach distance to ghost track significance
      trackGhostTrackWeight,			// weight of track participation in ghost track fit

      trackSumJetEtRatio,                       // ratio of track sum transverse energy over jet energy
      trackSumJetDeltaR,                        // pseudoangular distance between jet axis and track fourvector sum

      vertexCategory,                           // category of secondary vertex (Reco, Pseudo, No)

      jetNSecondaryVertices,                    // number of reconstructed possible secondary vertices in jet
      jetNSingleTrackVertices,                  // number of single-track ghost-track vertices

      vertexMass,                               // mass of track sum at secondary vertex
      vertexNTracks,                            // number of tracks at secondary vertex
      vertexFitProb,                            // vertex fit probability

      vertexEnergyRatio,                        // ratio of energy at secondary vertex over total energy
      vertexJetDeltaR,                          // pseudoangular distance between jet axis and secondary vertex direction

      flightDistance2dVal,                      // transverse distance between primary and secondary vertex
      flightDistance2dSig,                      // transverse distance significance between primary and secondary vertex
      flightDistance3dVal,                      // distance between primary and secondary vertex
      flightDistance3dSig,                      // distance significance between primary and secondary vertex

      trackSip2dValAboveCharm,                  // track 2D signed impact parameter of first track lifting mass above charm
      trackSip2dSigAboveCharm,                  // track 2D signed impact parameter significance of first track lifting mass above charm
      trackSip3dValAboveCharm,                  // track 3D signed impact parameter of first track lifting mass above charm
      trackSip3dSigAboveCharm,                  // track 3D signed impact parameter significance of first track lifting mass above charm

      neutralEnergy,                            // neutral ECAL clus. energy sum
      neutralEnergyOverCombinedEnergy,          // neutral ECAL clus. energy sum/(neutral ECAL clus. energy sum + pion tracks energy)
      neutralIsolEnergy,                        // neutral ECAL clus. energy sum in isolation band
      neutralIsolEnergyOverCombinedEnergy,      // neutral ECAL clus. energy sum in isolation band/(neutral ECAL clus. energy sum + pion tracks energy)
      neutralEnergyRatio,                       // ratio of neutral ECAL clus. energy sum in isolation band over neutral ECAL clus. energy sum
      neutralclusterNumber,                     // number of neutral ECAL clus.
      neutralclusterRadius,                     // mean DR between neutral ECAL clus. and lead.track

      leptonQuality,                            // lepton identification quality
      leptonQuality2,                           // lepton identification quality 2
      trackP0Par,                               // track momentum along the jet axis, in the jet rest frame
      trackP0ParRatio,                          // track momentum along the jet axis, in the jet rest frame, normalized to its energy"
      trackChi2,                                // track fit chi2
      trackNTotalHits,                          // number of valid total hits
      trackNPixelHits,                          // number of valid pixel hits

      algoDiscriminator,                        // discriminator output of an algorithm

      lastTaggingVariable
    };
  }

  // import only TaggingVariableName type into reco namespace
  using btau::TaggingVariableName;

  extern const char* TaggingVariableDescription[];
  extern const char* TaggingVariableTokens[];

  TaggingVariableName getTaggingVariableName ( const std::string & name );

  typedef float TaggingValue;
  
  // cannot use a const member since the STL containers relie on the default assignment operator
  // typedef std::pair< const TaggingVariableName, TaggingValue > TaggingVariable;
  typedef std::pair< TaggingVariableName, TaggingValue > TaggingVariable;

  struct TaggingVariableCompare {
    bool operator() (const TaggingVariable& i, const TaggingVariable& j) {
      return i.first < j.first;
    }

    bool operator() (const TaggingVariable& i, TaggingVariableName tag) {
      return i.first < tag;
    }

    bool operator() (TaggingVariableName tag, const TaggingVariable& i) {
      return tag < i.first;
    }

  };

  // implementation via std::vector where
  //  - m_list is kept sorted via stable_sort after each insertion
  //  - extraction is done via binary search
  class TaggingVariableList {
  public:
    TaggingVariableList() : m_list() { }
    TaggingVariableList( const TaggingVariableList& list ) : m_list( list.m_list ) { }

    // [begin, end) must identify a valid range of iterators to TaggingVariableList
    template <typename InputIterator>
    TaggingVariableList( const InputIterator begin, const InputIterator end ) : m_list() {
      BOOST_STATIC_ASSERT(( boost::is_convertible< const TaggingVariableList, typename boost::pointee<InputIterator>::type >::value ));
      for (const InputIterator i = begin; i != end; i++)
        insert(*i);
    }

    /**
     *  STL-like accessors 
     */
    typedef std::vector < TaggingVariable >::const_iterator const_iterator;
    typedef std::pair < const_iterator, const_iterator > range;
    size_t size() const { return m_list.size(); }
    const_iterator begin() const { return m_list.begin(); }
    const_iterator end() const { return m_list.end(); }
    void push_back ( const TaggingVariable & t ) { m_list.push_back ( t ); }

    ~TaggingVariableList() { }


  private:
    std::vector< TaggingVariable > m_list;
      
  public:
    bool checkTag( TaggingVariableName tag ) const;
    
    void insert( const TaggingVariable & variable, bool delayed = false );
    void insert( const TaggingVariableList & list );
    void insert( TaggingVariableName tag, TaggingValue value, bool delayed = false );
    void insert( TaggingVariableName tag, const std::vector<TaggingValue> & values, bool delayed = false );

    void finalize( void );
    
    TaggingValue get( TaggingVariableName tag ) const;
    TaggingValue get( TaggingVariableName tag, TaggingValue defaultValue ) const;
    std::vector<TaggingValue> getList( TaggingVariableName tag, bool throwOnEmptyList = true ) const;

    range getRange( TaggingVariableName tag ) const;

    TaggingValue operator[]( TaggingVariableName tag ) const {
      return get( tag );
    }
  };

  DECLARE_EDM_REFS( TaggingVariableList )

}

#endif // DataFormats_BTauReco_TaggingVariable_h
