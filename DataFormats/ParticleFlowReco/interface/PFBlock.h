#ifndef RecoParticleFlow_PFAlgo_PFBlock_h
#define RecoParticleFlow_PFAlgo_PFBlock_h 

#include <map>
#include <iostream>

/* #include "boost/graph/adjacency_matrix.hpp" */


// #include "DataFormats/ParticleFlowReco/interface/PFBlockLink.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
// #include "DataFormats/ParticleFlowReco/interface/PFBlockParticle.h"
#include "DataFormats/Common/interface/OwnVector.h"



namespace reco {

  
  /// \brief Block of elements
  /*!
    \author Colin Bernet
    \date January 2006

    A PFBlock is: 
    - a set of topologically connected elements.
    - a set of links between these elements
  */
  
  class PFBlock {


  public:

    struct Link {
      Link()   : distance(-1), test(0) {}
      Link(float d, char t) : distance(d), test(t) {}
      float distance;
      char test;
    };

    typedef edm::OwnVector< reco::PFBlockElement >::const_iterator IE;
    /*     typedef std::vector< reco::PFBlockLink >::const_iterator IL; */
    
    // typedef std::vector< std::vector<double> > LinkData;
    typedef std::map< unsigned int,  Link >  LinkData;
    
    enum LinkTest {
      LINKTEST_RECHIT,
      LINKTEST_NLINKTEST,
      LINKTEST_ALL
    };

    PFBlock() {}
    // PFBlock(const PFBlock& other);

    /// add an element to the current PFBlock
    /// the block will keep a copy.
    void addElement( reco::PFBlockElement* element );
    
    void bookLinkData();

    /// makes the correspondance between a 2d element matrix and 
    /// the 1D vector which is the most compact way to store the matrix
    bool matrix2vector(unsigned i, unsigned j, unsigned& index) const;

    /// set a link between elements of indices i1 and i2, of "distance" dist
    /// the link is set in the linkData vector provided as an argument.
    /// As indicated by the 'const' statement, 'this' is not modified.
    void setLink(unsigned i1, 
		 unsigned i2, 
		 double dist, 
                 LinkData& linkData, 
		 LinkTest  test=LINKTEST_RECHIT ) const;

    /// lock an element ( unlink it from the others )
    /// Colin: this function is misleading
    /// void lock(unsigned i, LinkData& linkData ) const;


    /// fills a map with the elements associated to element i.
    /// elements are sorted by increasing distance.
    /// if specified, only the elements of type "type" will be considered
    /// if specified, only the link calculated from a certain "test" will 
    /// be considered: distance test, etc..
    void associatedElements( unsigned i,
                             const LinkData& linkData, 
                             std::multimap<double, unsigned>& sortedAssociates,
                             reco::PFBlockElement::Type type = PFBlockElement::NONE,
			     LinkTest test=LINKTEST_RECHIT ) const; 
      

    /// \return distance of link
    double dist(unsigned ie1, 
		unsigned ie2, 
		const LinkData& linkData, 
		LinkTest  test ) const {
      return dist(ie1,ie2,linkData);
    }

    /// \return distance of link
    double dist( unsigned ie1, 
		 unsigned ie2, 
                 const LinkData& linkData) const;

    /// \return elements
    const edm::OwnVector< reco::PFBlockElement >& elements() const {
      return elements_;
    }

    /// \return link data
    const LinkData& linkData() const {
      return linkData_;
    }

    /// \return link data
    LinkData& linkData() {
      return linkData_;
    }

  private:
    
    /// \return size of linkData_, calculated from the number of elements
    unsigned linkDataSize() const;
    
    /// link data (permanent)
    LinkData                                        linkData_;
     
    /// all elements 
    edm::OwnVector< reco::PFBlockElement >          elements_;
        
  };

  std::ostream& operator<<( std::ostream& out, const PFBlock& co );

}

#endif


  
