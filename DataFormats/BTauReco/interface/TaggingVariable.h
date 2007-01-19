#ifndef DataFormats_BTauReco_TaggingVariable_h
#define DataFormats_BTauReco_TaggingVariable_h

#include <utility>
#include <vector>
#include <string>

#include <boost/static_assert.hpp>
#include <boost/pointee.hpp>
#include <boost/type_traits/is_convertible.hpp>

#include "DataFormats/BTauReco/interface/TaggingVariableFwd.h" 

namespace reco {

  namespace btag {

    // define the nuem in a namespace to avoid polluting reco with all the enum values
    enum TaggingVariableName {
      jetEnergy = 0,    // jet energy
      jetPt,            // jet transverse momentum
      jetEta,           // jet pseudorapidity
      jetPhi,           // jet polar angle
      trackMomemtum,    // track momentum
      trackEta,         // track pseudorapidity
      trackPhi,         // track polar angle
      trackSip2d,       // track 2D signed impact parameter significance
      trackSip3d,       // track 3D signed impact parameter significance
      trackPtRel,       // track transverse momentum, relative to the jet axis
      trackPpar,        // track parallel momentum, along the jet axis
      trackEtaRel,      // track pseudorapidity, relative to the jet axis
      trackDeltaR,      // track pseudoangular distance from the jet axis
      trackPtRatio,     // track transverse momentum, relative to the jet axis, normalized to its energy
      trackPparRatio,   // track parallel momentum, along the jet axis, normalized to its energy
      vertexCategory,   // category of secondary vertex (Reco, Pseudo, No)
      vertexMass,       // mass of secondary vertex
      vertexMultiplicity, // track multiplicity at secondary vertex
      flightDistance2DSignificance, // significance in 2d of distance between primary and secondary vtx
      eSVXOverE, // ratio of energy at secondary vertex over total energy
      trackSip2dAbCharm, // track 2D signed impact parameter significance above charm mass

      lastTaggingVariable
    };
  }

  // import only TaggingVariableName type into reco namespace
  using btag::TaggingVariableName;

  extern const char* TaggingVariableDescription[];
  extern const char* TaggingVariableTokens[];

  TaggingVariableName getTaggingVariableName ( const std::string & name );

  typedef double TaggingValue;
  
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
    size_t size() const { return m_list.size(); }
    const_iterator begin() const { return m_list.begin(); }
    const_iterator end() const { return m_list.end(); }
    void push_back ( const TaggingVariable & t ) { m_list.push_back ( t ); }

    ~TaggingVariableList() { }


  private:
    std::vector< TaggingVariable > m_list;
      
  public:
    bool checkTag( TaggingVariableName tag ) const;
    
    void insert( const TaggingVariable& variable );
    void insert( TaggingVariableName tag, const std::vector<TaggingValue> values );
    void insert( const TaggingVariableList& list );
    
    TaggingValue get( TaggingVariableName tag ) const;
    std::vector<TaggingValue> getList( TaggingVariableName tag ) const;

    TaggingValue operator[]( TaggingVariableName tag ) const {
      return get( tag );
    }
  };

}

#endif // DataFormats_BTauReco_TaggingVariable_h
