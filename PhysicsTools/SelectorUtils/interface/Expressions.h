#ifndef SelectorUtils_Expressions_h
#define SelectorUtils_Expressions_h

// expression templates
// Benedikt Hegner, DESY

include <iostream>

namespace reco{

// define expression objects
template<class A>
class DExpr {
  public:
    typedef typename A::ValType ValType;
    DExpr( const A& x = A() ): mA(x) {}
    double operator()( const ValType& x ) const { return mA(x); }
  private:
    A mA;
};


template<class ActON>
class DExprLiteral {
  public:
    typedef ActON ValType;
    DExprLiteral( double val ) { mVal = val; } 
    double operator()( const ValType& ) const { return mVal; }
  private:
    double mVal;
};


template<class ValType>
class BBase {
  public:
    BBase() {};
  virtual bool operator()(const ValType& x) const { return 1;}
    virtual BBase * clone() { return new BBase(*this);}
};

template<class A>
class BExpr : public BBase<typename A::ValType> {
  public:
    typedef typename A::ValType ValType;
    BExpr( const A& x = A() ): mA(x) {}
	bool operator()( const ValType& x ) const { return mA(x); }
  private:
    A mA;
};


// bool expression for functional/functional
template< class A, class Op, class B >
class BExprOp {
  public:
    typedef typename A::ValType ValType;
    BExprOp( const A &a, const B &b ): mA(a), mB(b) {}
    bool operator() (const ValType& x) const { return Op::apply( mA(x), mB(x) ); }
  private:
    A mA;
    B mB;  
};


// bool expression for double/functional
template< class A, class Op >
class BExprOpD {
  public:
	typedef typename A::ValType ValType;  
	BExprOpD( const A &a, const double b ): mA(a), iB(b) {}
    bool operator() (const ValType& x) const { return Op::apply( mA(x), iB ); }
  private:
    A mA;
    double iB;
};


// user defined methods
template<class ActON>
class UserFun {
  public:
    typedef ActON ValType;
    UserFun( double (ValType::* aMethod)() const ) { mpMethod = aMethod; }
    double operator()( const ValType& x ) const { return (x.*mpMethod)(); }
  private:
    double (ActON::* mpMethod) () const;
};


// implement the operators less, more, and
class Less {
public:
  static inline bool apply( double a, double b) { return (a<b);}
};

class More {
public:
  static inline bool apply( double a, double b) { return (a>b);}
};

class BAnd {
public:
  static inline bool apply( bool a, bool b ) { return ( a&&b );}
};



// OPERATOR SECTION

// functional < functional
template< class A, class B >
BExpr< BExprOp< DExpr<A> ,Less, DExpr<B> > >
operator<(const DExpr<A>& a, const DExpr<B>& b)
{
  typedef BExprOp< DExpr<A>, Less, DExpr<B> > Expr_t;
  return BExpr<Expr_t>(Expr_t(a, b));
}

// functional < double
template<class A>
BExpr< BExprOpD< DExpr<A> ,Less> >
operator<(const DExpr<A>& a, double b)
{
  typedef BExprOpD< DExpr<A>,Less> Expr_t;
  return BExpr<Expr_t>(Expr_t(a, b));
}

// double < functional
template<class B>
BExpr< BExprOpD< DExpr<B> ,Less> >
operator<(double a, const DExpr<B>& b)
{
  typedef BExprOpD< DExpr<B>,Less> Expr_t;
  return BExpr<Expr_t>(Expr_t(a, b));
}

// functional > functional
template< class A, class B >
BExpr< BExprOp< DExpr<A> ,More, DExpr<B> > >
operator>(const DExpr<A>& a, const DExpr<B>&b)
{
  typedef BExprOp< DExpr<A>, More, DExpr<B> > Expr_t;
  return BExpr<Expr_t>(Expr_t(a, b));
}

// functional > double
template<class A>
BExpr< BExprOpD< DExpr<A> ,More> >
operator>(const DExpr<A>& a, double b)
{
  typedef BExprOpD< DExpr<A>,More> Expr_t;
  return BExpr<Expr_t>(Expr_t(a, b));
}

// double > functional
template<class B>
BExpr< BExprOpD< DExpr<B> ,More> >
operator>(double a, const DExpr<B>& b)
{
  typedef BExprOpD< DExpr<B>,More> Expr_t;
  return BExpr<Expr_t>(Expr_t(a, b));
}

// boolean A and B
template< class A, class B >
BExpr< BExprOp< BExpr<A>, BAnd, BExpr<B> > >
operator&&( const BExpr<A>& a, const BExpr<B>& b )
{
  typedef BExprOp< BExpr<A>, BAnd, BExpr<B> > Expr_t;
  return BExpr<Expr_t>(Expr_t(a, b));
} 

// END OPERATOR SECTION


template<class Type>
class Selector {
  public:    
    Selector() {}
    template<class BExpr>
    void operator=(BExpr e){tmp = e.clone();} 
    void operator()(Type * aClass) {std::cout << tmp->operator()(*aClass) << std::endl;}  // for now just make a print to the screen
  private:
    BBase<Type>* tmp;  //@TODO: add exception handling in case of forgotten expression definition
};

} //namespace reco

#endif
