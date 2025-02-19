#ifndef HLTReco_HLTResult_h
#define HLTReco_HLTResult_h
#include <boost/static_assert.hpp>
#include <algorithm>

namespace reco {

  namespace hlt {

    template<unsigned int numberOfBits, typename word = unsigned short>
    struct wordConstants {
      enum {
	bytesInAWord = sizeof( word ),
	numberOfWords = 1 + ( numberOfBits - 1 ) / bytesInAWord
      };
    };

    template<unsigned short i, typename W>
    struct mask { 
      enum { 
	wordOffset = i / W::numberOfWords,
	bitOffset = i % W::numberOfWords,
	value = mask<bitOffset - 1, W>::value << 1 
      }; 
    };

    template<typename W>
    struct mask<0, W> { 
      enum {
	wordOffset = 0,
	bitOffset = 0,
	value = 1
      }; 
    };

  }

  template<unsigned int numberOfBits, typename word = unsigned short>
  class HLTResult {
    BOOST_STATIC_ASSERT( numberOfBits > 0 );
  public:
    HLTResult() { std::fill( words_, words_ + size, 0 ); }
    HLTResult( word w[] ) { std::copy( w, w + size, words_ ); }
  public:
    template<unsigned short i>
    bool match() const {
      typedef hlt::mask<i, wordConstants> mask;
      return words_[ mask::wordOffset ] & mask::value;
    }
    template<unsigned short i>
    void set() {
      typedef hlt::mask<i, wordConstants> mask;
      words_[ mask::wordOffset ] |= mask::value;
    }
    template<unsigned short i>
    void unSet() {
      typedef hlt::mask<i, wordConstants> mask;
      words_[ mask::wordOffset ] &= ! mask::value;
    }
    
  private:
    typedef hlt::wordConstants<numberOfBits, word> wordConstants;
    enum { wordSize = sizeof( word ), size = 1 + ( numberOfBits - 1 ) / wordSize };
    word words_[ size ];
  };

}

#endif
