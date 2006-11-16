/// Jet counts
template <int nBits>
class L1GctJetCount : public L1GctUnsignedInt<nBits> {

 public:

  /// Construct a counter and initialise its value to zero
  L1GctJetCount();
  /// Construct a counter, checking for overFlow 
  L1GctJetCount(unsigned value);
  /// Destructor
  ~L1GctJetCount();

  /// Copy contructor to move data between representations with different numbers of bits
  template <int mBits>
  L1GctJetCount(const L1GctJetCount<mBits>& rhs);

  /// Set value from unsigned
  void setValue(unsigned value);

  /// set the overflow bit
  void setOverFlow(bool oflow);

  /// Define increment operators, since this is a counter.
  L1GctJetCount& operator++ ();
  L1GctJetCount operator++ (int);

  /// add two numbers
  L1GctJetCount operator+ (const L1GctJetCount &rhs) const;

  /// overload = operator
  L1GctJetCount& operator= (int value);

};

template <int nBits>
L1GctJetCount<nBits>::L1GctJetCount() : L1GctUnsignedInt<nBits>() {}

template <int nBits>
L1GctJetCount<nBits>::L1GctJetCount(unsigned value) : L1GctUnsignedInt<nBits>(value) {}

template <int nBits>
L1GctJetCount<nBits>::~L1GctJetCount() {}

// copy contructor to move data between
// representations with different numbers of bits
template <int nBits>
template <int mBits>
L1GctJetCount<nBits>::L1GctJetCount(const L1GctJetCount<mBits>& rhs) {
  m_nBits = nBits>0 && nBits<MAX_NBITS ? nBits : 16 ;
  this->setValue( rhs.value() );
  this->setOverFlow( this->overFlow() || rhs.overFlow() );
}

template <int nBits>
void L1GctJetCount<nBits>::setValue(unsigned value)
{
  // check for overflow
  if (value >= (static_cast<unsigned>((1<<m_nBits) - 1)) ) {
    m_overFlow = true;
    m_value = ((1<<m_nBits) - 1);
  } else {
    m_value = value;
  }

}

template <int nBits>
void L1GctJetCount<nBits>::setOverFlow(bool oflow)
{
  m_overFlow = oflow;
  if (oflow) { m_value = ((1<<m_nBits) - 1); }
}

// increment operators
template <int nBits>
L1GctJetCount<nBits>&
L1GctJetCount<nBits>::operator++ () {

  this->setValue(m_value+1);
  return *this;
}

template <int nBits>
L1GctJetCount<nBits>
L1GctJetCount<nBits>::operator++ (int) {

  L1GctJetCount<nBits> temp(m_value);
  temp.setOverFlow(m_overFlow);
  this->setValue(m_value+1);
  return temp;
}

// add two jet counts
template <int nBits>
L1GctJetCount<nBits>
L1GctJetCount<nBits>::operator+ (const L1GctJetCount<nBits> &rhs) const {

  // temporary variable for storing the result (need to set its size)
  L1GctJetCount<nBits> temp;

  unsigned sum;
  bool ofl;

  // do the addition here
  sum = this->value() + rhs.value();
  ofl = this->overFlow() || rhs.overFlow();

  //fill the temporary argument
  temp.setValue(sum);
  temp.setOverFlow(temp.overFlow() || ofl);

  // return the temporary
  return temp;

}

// overload assignment by int
template <int nBits>
L1GctJetCount<nBits>& L1GctJetCount<nBits>::operator= (int value) {
  
  this->setValue(value);
  return *this;

}

// overload ostream<<
template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctTwosComplement<nBits>& data) {

  s << "L1GctTwosComplement raw : " << data.raw() << ", " << "value : " << data.value();
  if (data.overFlow()) { s << " Overflow set! "; }

  return s;

}

template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctUnsignedInt<nBits>& data) {

  s << "L1GctUnsignedInt value : " << data.value();
  if (data.overFlow()) { s << " Overflow set! "; }

  return s;

}

template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctJetCount<nBits>& data) {

  s << "L1GctJetCount value : " << data.value();
  if (data.overFlow()) { s << " Overflow set! "; }

  return s;

}


/// typedef for the data type used for final output jet counts
typedef L1GctJetCount<5>        L1GctJcFinalType;
/// typedef for the data type used for Wheel card jet counts
typedef L1GctJetCount<3>        L1GctJcWheelType;


