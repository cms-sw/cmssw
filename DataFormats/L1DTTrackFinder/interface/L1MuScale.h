//-------------------------------------------------
//
//   \class L1MuScale
//
//   Description:  A Scale for use in the L1 muon trigger
//                 
//                  
//                
//   $Date: 2006/06/01 00:00:00 $
//   $Revision: 1.1 $ 
//
//   Author :
//   Hannes Sakulin      HEPHY / Vienna
//
//--------------------------------------------------
#ifndef L1MU_SCALE_H
#define L1MU_SCALE_H

#include <iostream>
#include <vector>
#include "DataFormats/L1DTTrackFinder/interface/L1MuPacking.h"

/**
 * \class L1MuScale
 *
 * define the abstract interface to all scales
*/

class L1MuScale {
 public:
  /// get the center of bin represented by packed
  virtual float getCenter(unsigned packed) const = 0;

  /// get the low edge of bin represented by packed
  virtual float getLowEdge(unsigned packed) const = 0; 

  /// get the upper edge of bin represented by packed
  virtual float getHighEdge(unsigned packed) const = 0;
  
  /// get the upper edge of the last bin
  virtual float getScaleMax() const = 0;

  /// get the lower edge of the first bin
  virtual float getScaleMin() const = 0;
  
  /// pack a value
  virtual unsigned getPacked(float value) const = 0;
 private:
};

//
// define default scale implementations
//


/**
 * \class L1MuBinnedScale
 *
 * implements a continuous scale of NBins bins. 
 *
 * the index into the binned scale runs from 0 to NBins-1.
 * 
 * It is packed into a data word (unsigned) using a Packing (template parameter)
 *  
 * If the packing accepts negative values, an offset can be specified which 
 * will be subtracted from the index before packing. ( The offset is in turn added to the packed
 * value before using it as an index into the scale.)
 *
*/

template < class thePacking>
class L1MuBinnedScale : public L1MuScale, public thePacking {
 public:

  ///
  /// constructor
  ///
  /// NBins=number of bins, Scale[NBins+1]=bin edges, idx_offset=offeset to index (if stored as signed)
  ///
  L1MuBinnedScale(int NBins, const float* Scale, int idx_offset=0) {
    m_NBins = NBins;
    m_idxoffset = idx_offset;

    m_Scale.reserve(m_NBins + 1);
    for (int i=0; i<m_NBins + 1; i++) 
      m_Scale[i] = Scale[i];
  };

  ///
  /// constructor 
  ///
  /// NBins=number of bins, xmin = low edge of first bin, 
  /// xmax=high edge of last bin, idx_offset=offeset to index (if stored as signed)
  ///
  L1MuBinnedScale(int NBins, float xmin, float xmax, int idx_offset=0) {
    m_NBins = NBins;
    m_idxoffset = idx_offset;

    m_Scale.reserve(m_NBins + 1);
    for (int i=0; i<m_NBins + 1; i++) 
      m_Scale[i] = xmin + i * (xmax-xmin) / m_NBins;
  };

  /// destructor
  virtual ~L1MuBinnedScale() {};

 
  /// get the center of bin represented by packed
  virtual float getCenter(unsigned packed) const {
    int idx = get_idx(packed);
    return (m_Scale[idx] + m_Scale[idx+1] )/ 2.;    
  };

  /// get the low edge of bin represented by packed
  virtual float getLowEdge(unsigned packed) const{
    return m_Scale[get_idx(packed)];
  };

  /// get the upper edge of bin represented by packed
  virtual float getHighEdge(unsigned packed) const{
    return m_Scale[get_idx(packed)+1];
  };
  
  /// pack a value
  virtual unsigned getPacked(float value) const {
    if (value < m_Scale[0] || value > m_Scale[m_NBins]) 
      cout << "***Error: L1MuBinnedScale::getPacked: value out of scale range: " << value << endl;
    int idx = 0;
    if (value < m_Scale[0]) idx=0;
    else if (value >= m_Scale[m_NBins]) idx = m_NBins-1;
    else {
      for (; idx<m_NBins; idx++) 
        if (value >= m_Scale[idx] && value < m_Scale[idx+1]) break;
    }

    return packedFromIdx(idx-m_idxoffset);
  };

  /// get the upper edge of the last bin
  virtual float getScaleMax() const { return m_Scale[m_NBins]; }

  /// get the lower edge of the first bin
  virtual float getScaleMin() const { return m_Scale[0]; }

 protected:
  int get_idx (unsigned packed) const {
    int idx = idxFromPacked( packed ) + m_idxoffset;
    if (idx<0) idx=0;
    if (idx>=m_NBins) idx=m_NBins-1;
    return idx;
  }

  int m_NBins;
  int m_idxoffset;
  vector<float> m_Scale;
};

/**
 * \class  L1MuSymmetricBinnedScale
 *
 * In the GMT the concept of a symmetric scale exists 
 * The internal representation of scale values is "pseudo-signed", i.e.
 * the highest bit stores the sign and the lower bits contain the absolute value
 * 
 * Attention: for reasons of symmetry, the low edge in this scale is the edge closer to zero.
 *            the high edge is the edge further away from zero
*/

template <int NBits>
class L1MuSymmetricBinnedScale : public L1MuScale, public L1MuPseudoSignedPacking<NBits> {
 public:
  
  ///
  /// constructor 
  ///
  /// NBins=number of bins (in one half of the scale), Scale[NBins+1]=bin edges
  ///
  L1MuSymmetricBinnedScale(int NBins, const float* Scale) {
    m_NBins = NBins;
    m_Scale.reserve(m_NBins + 1);
    for (int i=0; i<m_NBins + 1; i++) 
      m_Scale[i] = Scale[i];
  };

  ///
  /// constructor 
  ///
  /// NBins=number of bins, xmin = low edge of first bin (in positive half) 
  /// xmax=high edge of last bin (in positive half)
  ///
  L1MuSymmetricBinnedScale(int NBins, float xmin, float xmax) {
    m_NBins = NBins;
    m_Scale.reserve(m_NBins + 1);
    for (int i=0; i<m_NBins + 1; i++) 
      m_Scale[i] = xmin + i * (xmax-xmin) / m_NBins;
  };

  /// destructor
  virtual ~L1MuSymmetricBinnedScale() {};

  /// get the center of bin represented by packed
  virtual float getCenter(unsigned packed) const {
    int absidx = abs ( idxFromPacked( packed ) );
    if (absidx>=m_NBins) absidx=m_NBins-1;
    float center = (m_Scale[absidx] + m_Scale[absidx+1] )/ 2.;    
    float fsign = signFromPacked( packed ) == 0 ? 1. : -1.;
    return center * fsign;
  };

  /// get the low edge of bin represented by packed
  virtual float getLowEdge(unsigned packed) const{ // === edge towards 0 
    int absidx = abs ( idxFromPacked( packed ) );
    if (absidx>=m_NBins) absidx=m_NBins-1;
    float low = m_Scale[absidx];    
    float fsign = signFromPacked( packed ) == 0 ? 1. : -1.;
    return low * fsign;
  };

  /// get the upper edge of bin represented by packed
  virtual float getHighEdge(unsigned packed) const{
    cout << "Error: L1MuSymmetricBinnedScale::getHighEdge not implemented" << endl;
    return 0;
  };

  /// pack a value
  virtual unsigned getPacked(float value) const {
    float absval = fabs ( value );
    if (absval < m_Scale[0] || absval > m_Scale[m_NBins]) 
      cout << "***Error: L1MuSymmetricBinnedScale::getPacked: value out of scale range!!! abs(val) = " 
	   << absval << " min= " << m_Scale[0] << " max = " << m_Scale[m_NBins] << endl;
    int idx = 0;
    for (; idx<m_NBins; idx++) 
      if (absval >= m_Scale[idx] && absval < m_Scale[idx+1]) break;
    if (idx >= m_NBins) idx = m_NBins-1;
    return packedFromIdx(idx, (value>=0) ? 0 : 1);
  };
  /// get the upper edge of the last bin (posivie half)
  virtual float getScaleMax() const { return m_Scale[m_NBins]; }

  /// get the lower edge of the first bin (positive half)
  virtual float getScaleMin() const { return m_Scale[0]; }
 protected:
  int m_NBins;
  vector<float> m_Scale;
};
#endif









































