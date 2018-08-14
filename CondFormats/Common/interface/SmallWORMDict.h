#ifndef cond_SmallWORMDict_h
#define cond_SmallWORMDict_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include<vector>
#include<string>
#include <boost/iterator_adaptors.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

// Function for testing SmallWORMDict
namespace test {
  namespace SmallWORMDict {
    int test();
  }
}

namespace cond {

/** A small WORM Dictionary of small words
    optimized to do a single allocation
 */

  class SmallWORMDict {
    friend int test::SmallWORMDict::test();

  public:
    SmallWORMDict();
    ~SmallWORMDict();
    
    struct Frame {
      Frame(): b(nullptr){}
      Frame(char const * ib,
	    unsigned int il,
	    unsigned int iind) :
	b(ib),l(il),ind(iind){}
      char const * b;
      unsigned int l;
      unsigned int ind;
    };

    struct IterHelp {
      typedef Frame result_type;
      IterHelp() : v(nullptr){}
      IterHelp(SmallWORMDict const & iv) : v(&iv){}
      
      result_type const & operator()(int i) const {
	int k = (0==i) ? 0 : v->m_index[i-1]; 
	return  frame(&v->m_data[k], v->m_index[i]-k, i);
      } 
      
      Frame const & frame(char const * b,
			  unsigned int l,
			  unsigned int ind) const { 
	f.b = b; f.l=l; f.ind=ind;
	return f;
      }
      
    private:
      SmallWORMDict const * v;
      mutable Frame f;
    };
    
    friend struct IterHelp;

    typedef boost::transform_iterator<IterHelp,boost::counting_iterator<int> > const_iterator;

 
    const_iterator begin() const {
      return  boost::make_transform_iterator(boost::counting_iterator<int>(0),
					     IterHelp(*this));
    }
    
    const_iterator end() const {
      return  boost::make_transform_iterator(boost::counting_iterator<int>(size()),
					     IterHelp(*this));
    }

    Frame operator[](int i) const {
      int k = (0==i) ? 0 : m_index[i-1]; 
      return  Frame(&m_data[k], m_index[i]-k, i);
      } 

    const_iterator find(std::string const & s) const;

    const_iterator find(char const * s) const;

    // constructror from config
    explicit SmallWORMDict(std::vector<std::string> const & idict);
    
    // find location of a word
    size_t index(std::string const & s) const;

    size_t index(char const * s) const;

    size_t size() const;

  private: 
    std::vector<char> m_data;
    std::vector<unsigned int> m_index;
  
 COND_SERIALIZABLE;
};


}

#endif
