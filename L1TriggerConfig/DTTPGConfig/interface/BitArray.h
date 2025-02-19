//-------------------------------------------------
//
//   Class: BitArray
//
//   Description: Manipulates arrays of bits
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//
//
//--------------------------------------------------
#ifndef BIT_ARRAY_H_
#define BIT_ARRAY_H_



//---------------
// C++ Headers --
//---------------
#include <cassert>
#include <iostream>

//              ---------------------
//              -- Class Interface --
//              ---------------------

template<int N>
class BitArray {

 public:
  class refToBit;
  friend class refToBit;

  // reference to bit class
  class refToBit{
    friend class BitArray;

    //refToBit();

  public:
    refToBit() {}
    refToBit(BitArray& b, int pos) {
      _word = &b.getWord(pos);
      _pos = getPosInWord(pos);
    }
    ~refToBit() {}

    refToBit& operator=(const int val) {
      if(val) {
	*_word |= getPosMask(_pos);
      } else {
	*_word &= ~(getPosMask(_pos));
      }
      return *this;
    }

    refToBit& operator=(const refToBit& rtb) {
      if( (*(rtb._word) & getPosMask(rtb._pos)) ) {
	*_word |= getPosMask(_pos);
      } else { 
	*_word &= ~getPosMask(_pos);
      }
      return *this;
    }

    int operator~() const { return ((*_word)&getPosMask(_pos))==0; }

    operator int() const { return ((*_word)&getPosMask(_pos))!=0; }

    refToBit& flip() {
      *_word^=getPosMask(_pos);
      return *this;
    }
    
  private:
    unsigned* _word;
    int _pos;

  };

  BitArray() {this->zero();}

  BitArray(const BitArray<N>& br) {
    for (int i=0;i<this->nWords();i++) {
      _data[i] = br._data[i];         
    }
    this->cleanUnused();
  }
  BitArray(const char* str) { 
    this->zero(); 
    this->assign(0,this->nBits(),str); 
    this->cleanUnused();
  }
  BitArray(const char* str, const int p, const int n) {
    this->zero(); 
    this->assign(p, n, str);
  }
  BitArray(const unsigned i) { 
    this->zero();
    _data[0] = i;                 // the nBit least sign. bits are considered
    this->cleanUnused();
  }
/*
  BitArray(const unsigned long i) { 
    this->zero();
    unsigned j = i&static_cast<unsigned long>(0xffffffff);
    _data[0] = j;                 // the nBit least sign. bits are considered
    if(this->nBits()>32) {
      j=i>>32;
      _data[1] = j;
    }
    this->cleanUnused();
  }
*/
  //  Destructor 
  // ~BitArray() {}
  
  // Return number of bits
  inline int nBits() const { return N; }
  inline int size() const { return this->nBits(); }
  
  // Return number of words
  inline int nWords() const { return N/32+1; }
  
  // Return a data word
  unsigned dataWord(const int i) const {
    assert(i>=0 && i<this->nWords());
    return _data[i];
  }
  unsigned & dataWord(const int i) {
    assert(i>=0 && i<this->nWords());
    return _data[i];
  }

  // Return the dataword which contains bit in position
  unsigned & getWord(const int pos) {
    assert(pos>=0&& pos<this->nBits());
    return _data[pos/32];
  }
  unsigned getWord(const int pos) const {
    assert(pos>=0&& pos<this->nBits());
    return _data[pos/32];
  }

  // Return the position inside a word given the position in bitArray
  static int getPosInWord(const int pos) {
    // assert(pos>=0&& pos<this->nBits());
    return pos%32;
  }
  
  // Return the word mask for a given bit
  static unsigned getPosMask(const int pos) {
    return static_cast<unsigned>(1)<<getPosInWord(pos);
  }

  // how many bits are not used (most significative bits of last word)
  int unusedBits() const {
    if(this->nBits()==0)return 32;
    return 31-((this->nBits()-1)%32);
  }
 
  // mask to get rid of last word unused bits
  unsigned lastWordMask() const {
    return static_cast<unsigned>(0xffffffff)>>(this->unusedBits());
  }

  // set unused bits to 0
  void cleanUnused() {
    _data[this->nWords()-1] &= (this->lastWordMask());
  }
 
  // count non 0 bits
  int count() const {
    int n=0;
    for(int i=0;i<this->nBits();i++) {
      if(this->element(i))n++;
    }
    return n;
  }
 
  // true if any bit == 1
  int any() {
    int nw = this->nWords();
    int ub = unusedBits();
    if(this->dataWord(nw-1)<<ub!=0)return 1;
    if(nw>1){
      for (int iw=0;iw<nw-1;iw++){
	if(this->dataWord(iw)!=0) return 1;
      }
    }
    return 0;
  }    
 
  // true if any bit == 0
  int none() {
    int nw = this->nWords();
    int ub = unusedBits();
    if(this->dataWord(nw-1)<<ub!=0xffffffff)return 1;
    if(nw>1){
      for (int iw=0;iw<nw-1;iw++){
	if(this->dataWord(iw)!=0xffffffff) return 1;
      }
    }
    return 0;
  }    
 
  // Return i^th elemnet
  int element(const int pos) const {
    return (getWord(pos)&getPosMask(pos))!=static_cast<unsigned>(0);
  }
  inline int test(const int i) const { return element(i); }

  // Set/unset all elements
  void zero() {
    for (int i=0;i<this->nWords();i++) {
      _data[i] = 0x0;                // set to 0
    }
  }
  inline void reset() { zero(); }

  void one() {
    for (int i=0;i<this->nWords();i++) {
      _data[i] = 0xffffffff;       // set to 1
    }
  }
  
  // Set/unset i^th element
  void set(const int i)  { getWord(i) |= getPosMask(i); }
  void unset(const int i) { getWord(i) &= ~getPosMask(i); }
  inline void reset(const int i) { this->unset(i); }
  
  // Set the i^th element to a given value
  inline void set(const int i, const int val) { this->assign(i,1,val); }
  inline void set(const int i, const char* str) { this->assign(i,1,str); }

  // Set/unset many bits to a given integer/bitArray/string
  void assign(const int p, const int n, const int val) {
    assert(p>=0 && p+n<=this->nBits());  
    // only the n least significant bits of val are considered
    for(int i=0; i<n;i++){
      if(val>>i&1) {
	this->set(p+i);
      } else {
	this->unset(p+i);
      }
    }
  }
  void assign(const int p, const int n, const BitArray<N>& val) {
    assert(p>=0 && p+n<=this->nBits());  
    // only the n least significant bits of val are considered
    for(int i=0; i<n;i++){
      if(val.element(i)) {
	this->set(p+i);
      } else {
	this->unset(p+i);
      }
    }
  }
  void assign(const int p, const int n, const char* str) {
    assert(p>=0 && p+n<=this->nBits());  
    // only the n least significant bits of val are considered
    for(int i=0; i<n;i++){
      assert(str[i]=='1'||str[i]=='0');  
      if(str[i]=='1') {
	this->set(p+n-i-1);   // reading a string from left to right 
      } else {                // --> most significative bit is the one 
	this->unset(p+n-i-1); // with lower string index
      }
    }
  }
      
  // Read a given range in an unsigned integer 
  unsigned read(const int p, const int n) const {
    assert(p>=0 && p+n<=this->nBits());  
    assert(n<=32); // the output must fit in a 32 bit word
    // only the n least significant bits of val are considered
    unsigned out=0x0;
    for(int i=0; i<n;i++){
      if(this->test(p+i)) out |= 1<<i;
    }
    return out;
  }

  // Read BitArray in bytes. Returns a BitArray<8>
  BitArray<8> byte(const int i) const {
    BitArray<8> out;
    if(i>=0&&i<4*this->nWords()){
      unsigned k=(_data[i/4]>>8*(i%4))&0xff;
      out=k;
    }
    return out;
  }
  // Assignment
  BitArray<N>& operator=(const BitArray<N>& a) {
    if(this != &a) {
      for (int i=0;i<this->nWords();i++) {
	_data[i] = a._data[i];
      }
    }
    this->cleanUnused();
    return *this;
  }
  
  // Conversion from unsigned
  BitArray<N>& operator=(const unsigned i) {
    this->zero();
    _data[0] = i;                 // the nBit least sign. bits are considered
    this->cleanUnused();
    return *this;
  }
/*    
  // Conversion from unsigned long
  BitArray<N>& operator=(const unsigned long i) {
    this->zero();
    unsigned j = i&0xffffffff;
    _data[0] = j;                 // the nBit least sign. bits are considered
    if(this->nBits()>32) {
      j=i>>32;
      _data[1] = j;
    }
    this->cleanUnused();
    return *this;
  }
*/    
  // Conversion from char
  BitArray<N>& operator=(const char* str) {
    this->zero();
    for(int i=0; i<this->nBits();i++){
      assert(str[i]=='1'||str[i]=='0');  
      if(str[i]=='1') {
	this->set(this->nBits()-i-1);   // reading a string from left to right 
      } else if(str[i]=='0') {    // --> most significative bit is the one 
	this->unset(this->nBits()-i-1); // with lower string index
      } else {
        break;                    // exit when find a char which is not 0 or 1
      }
    }
    this->cleanUnused();
    return *this;
  }
    
  // Print
  std::ostream & print(std::ostream& o=std::cout) const {
    for(int i = this->nBits()-1; i>=0; i--){
      o << this->element(i);
    }
    return o;
  }
  
  // direct access to set/read elements
  refToBit operator[](const int pos) { return refToBit(*this,pos); }
  int operator[](const int pos) const { return element(pos); }
  
  // logical operators ==
  bool operator==(const BitArray<N>& a) const {
   int nw = this->nWords();
    int ub = this->unusedBits();
    if(this->dataWord(nw-1)<<ub!=         // check last word
           a.dataWord(nw-1)<<ub)return 0;
    if(nw>1){
      for (int iw=0;iw<nw-1;iw++){
	if(this->dataWord(iw)!=a.dataWord(iw)) return 0;
      }
    }
    return 1;
  }
  
  // logical operators <
  bool operator<(const BitArray<N>& a) const {
    int nw = this->nWords();
    int ub = this->unusedBits();
    unsigned aaa = this->dataWord(nw-1)<<ub; // ignore unused bits
    unsigned bbb =     a.dataWord(nw-1)<<ub; // in both operands
    if        (aaa<bbb) {
      return 1;
    } else if (aaa>bbb) {
      return 0;
    }
    if(nw>1){
      for (int iw=nw-2;iw>=0;iw--){
	if        (this->dataWord(iw)<a.dataWord(iw)) {
	  return 1;
	} else if (this->dataWord(iw)>a.dataWord(iw)) {
	  return 0;
	}
      }
    }
    return 0;
  }
  
  // logical operators !=
  bool operator!=(const BitArray<N>& a) const { return !(a==*this); }
  
  // logical operators >=
  bool operator>=(const BitArray<N>& a) const{ return !(*this<a); }
  
  // logical operators >
  bool operator>(const BitArray<N>& a) const { return !(*this<a||*this==a); }
  
  // logical operators <=
  bool operator<=(const BitArray<N>& a) const { return !(*this>a); }
  
  // non-const bit by bit negation
  BitArray<N>& flip () {
    for(int i=0;i<this->nWords();i++) {
      _data[i] = ~_data[i];
    }
    return *this;
  }

  // const bit by bit negation
  BitArray<N> operator~ () const { return BitArray<N> (*this).flip(); }

  // bit by bit AND and assignment
  BitArray<N>& operator&= (const BitArray<N>& a) {
    for(int i = 0;i<this->nWords();i++) {
      this->dataWord(i) &= a.dataWord(i);
    }
    return *this;
  }    

  // bit by bit AND
  BitArray<N> operator&(const BitArray<N>& a) {return BitArray<N> (*this)&=a; }
  
  // bit by bit OR and assignment
  BitArray<N>& operator|=(const BitArray<N>& a) {
    for(int i = 0;i<this->nWords();i++) {
      this->dataWord(i) |= a.dataWord(i);
    }
    return *this;
  }    

  // bit by bit AND
  BitArray<N> operator|(const BitArray<N>& a) {return BitArray<N> (*this)|=a;}
  
  // bit by bit XOR and assignment
  BitArray<N>& operator^=(const BitArray<N>& a) {
    for(int i = 0;i<this->nWords();i++) {
      this->dataWord(i) ^= a.dataWord(i);
    }
    return *this;
  }    

  // bit by bit XOR
  BitArray<N> operator^(const BitArray<N>& a) {return BitArray<N> (*this)^=a; }
  
  // left shift and assignment
  BitArray<N>& operator<<=(const int n) {
    assert(n>=0&&n<this->nBits());
    if(n==0)return *this;
    int i=0;
    for(i=this->nBits()-1;i>=n;i--) this->set(i,this->element(i-n));
    for(i=n-1;i>=0;i--) this->unset(i);
    return *this;
  }

  // left shift 
  BitArray<N> operator<<(const int n) { return BitArray<N> (*this)<<=n; }

  // right shift and assignment
  BitArray<N>& operator>>=(const int n) {
    assert(n>=0&&n<this->nBits());
    if(n==0)return *this;
    int i=0;
    for(i=0;i<this->nBits()-n;i++) this->set(i,this->element(i+n));
    for(i=this->nBits()-n;i<this->nBits();i++) this->unset(i);
    return *this;
  }

  // right shift
  BitArray<N> operator>>(const int n) { return BitArray<N> (*this)>>=n; }

  // sum and assignment
  BitArray<N>& operator+=(const BitArray<N>& a) {
    int rep=0;
    int sum=0;
    for(int i=0;i<this->nBits();i++) {
      sum=this->element(i)^rep;
      rep=this->element(i)&rep;
      this->set(i,sum^a.element(i));
      rep|=(sum&a.element(i));
    }
    return *this;
  }

  // sum
  BitArray<N> operator+(const BitArray<N>& a) {return BitArray<N> (*this)+=a; }

  // postfix increment
  BitArray<N>& operator++(int) {
    int i = 0;
    while(i<this->nBits()&&this->element(i)==1) { this->unset(i); i++; }
    if(i<this->nBits())this->set(i);
    return *this;
  }

  // const 2 complement
  BitArray<N> twoComplement() const { return BitArray<N> (~*this)++; }

  // non-const 2 complement
  BitArray<N>& twoComplement() { 
    (*this).flip();
    (*this)++;
    return *this;
  }

  // subtract and assignment
  BitArray<N>& operator-=(const BitArray<N>& a) {
    return *this+=a.twoComplement();
  }

  // subtract
  BitArray<N> operator-(const BitArray<N>& a) {return BitArray<N> (*this)-=a; }

private:
  
  unsigned _data[N/32+1];
};

/*
template<int N>
ostream & operator <<(ostream & o, const BitArray<N> &b) {
  b.print(o); return o;
}
*/

#endif
