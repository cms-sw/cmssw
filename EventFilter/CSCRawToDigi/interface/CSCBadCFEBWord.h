#ifndef CSCBadCFEBWord_h
#define CSCBadCFEBWord_h

/**
 * When a time slice is bad, it only has four words, and they all start with "B"
 */
#include<iosfwd>

class CSCBadCFEBWord {
public:
  /// make sure it really does start with a "B"
  bool check() const {return b_==0xb;}
  bool isBad() const {return true;}
  friend std::ostream & operator<<(std::ostream & os, const CSCBadCFEBWord &);
private:
  unsigned short word1_:4;
  unsigned short word2_:4;
  unsigned short zero_:1;
  unsigned short code_:3;
  unsigned short b_:4;
};


#endif
