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

    typedef edm::OwnVector< reco::PFBlockElement >::const_iterator IE;
    /*     typedef std::vector< reco::PFBlockLink >::const_iterator IL; */
    
    
    PFBlock() {}
    // PFBlock(const PFBlock& other);

    /// add an element to the current PFBlock
    /// the block will keep a copy.
    void addElement( reco::PFBlockElement* element );
    
    void bookLinkData();

    /// makes the correspondance between a 2d element matrix and 
    /// the 1D vector which is the most compact way to store the matrix
    bool matrix2vector(unsigned i, unsigned j, unsigned& index) const;

    /// set a link between elements of indices i1 and i2, of "distance" chi2
    /// the link is set in the linkData vector provided as an argument.
    void setLink(unsigned i1, unsigned i2, double chi2, 
                 std::vector<double>& linkData ) const;


    /// lock an element ( unlink it from the others )
    void lock(unsigned i, std::vector<double>& linkData ) const;


    /// fills a map with the elements associated to element i.
    /// elements are sorted by increasing chi2.
    /// if specified, only the elements of type "type" will be considered
    void associatedElements( unsigned i,
                             const std::vector<double>& linkData, 
                             std::map<double, unsigned>& sortedAssociates,
                             reco::PFBlockElement::Type type = PFBlockElement::NONE) const;  
      

    /// \return chi2 of link
    double chi2( unsigned ie1, unsigned ie2, 
                 const std::vector<double>& linkData ) const;

    /// \return elements
    const edm::OwnVector< reco::PFBlockElement >& elements() const 
      {return elements_;}

    /// \return link data
    const std::vector< double >& linkData() const 
      {return linkData_;}

    /// \return link data
    std::vector< double >& linkData()  
      {return linkData_;}

    friend std::ostream& operator<<( std::ostream& out, const PFBlock& co );

  private:
    
    /// \return size of linkData_, calculated from the number of elements
    unsigned linkDataSize() const;
    
    /// link data (permanent)
    std::vector< double >                           linkData_;
     
    /// all elements 
    edm::OwnVector< reco::PFBlockElement >          elements_;
        
  };
}

#endif


  
