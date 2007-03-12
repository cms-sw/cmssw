#ifndef SelectorUtils_Expressions_h
#define SelectorUtils_Expressions_h

// expression templates
// Benedikt Hegner, DESY

namespace reco{

template<class T>
class ExprBase {
  public:
    ExprBase() {};
    virtual bool operator()(const T& x) const { return 1;}
    virtual ExprBase * clone() { return new ExprBase(*this);}
};



template<class Type>
class Selector {
  public:    
    Selector() {}
    template<class Expr>
    void operator=(Expr e){tmp = e.clone();} 
    void operator()(Type * aClass) {return tmp->operator()(*aClass);}
  private:
    ExprBase<Type>* tmp;  //@TODO: add exception handling in case of forgotten expression definition
};


//the implementation part

template<class AnExpr>
class Expr : public ExprBase<typename AnExpr::RetType> {
  public:
    typedef typename AnExpr::ArgType ArgType;
    typedef typename AnExpr::RetType RetType;

    Expr( const AnExpr& expr = AnExpr() ):mExpr(expr) {};
    RetType operator()( const ArgType& x ) const
      { return mExpr(x); }

  private:
    AnExpr mExpr;
};


//hold expression literals
template< class Value, class ActON >
class ExprLiteral
{
public:
  typedef Value RetType;
  typedef ActON ArgType;

  ExprLiteral( const RetType& val)
  { mVal = val; } 

  RetType operator()( const ArgType& ) const
  { return mVal; }

private:
  RetType mVal;
};


/// hold basic identities
template < class RETType >
class ExprIdentity
{
public:
  typedef RETType RetType;
  typedef RETType ArgType;

  RetType operator()( const ArgType& x ) const
  { return x; }

};


template< class Value, class ActON >
class ExprUserFun
{
public:
  typedef Value RetType;
  typedef ActON ArgType;

  ExprUserFun( Value (ActON::* aFun)() const )
    {mFun = aFun;};

  RetType operator()( const ArgType& x ) const
    {return (x.*mFun)();}

private:
  RetType (ActON::* mFun)() const;
};


///////////////
// operators //
///////////////
template< class T >
struct Add {
  typedef T RetType;
  static inline RetType apply( const RetType& a, const RetType& b )
    {return ( a + b );}
};

template< class T >
struct Sub {
  typedef T RetType;
  static inline RetType apply( const RetType& a, const RetType& b )
    {return ( a - b );}
};

template< class T >
struct Mul {
  typedef T RetType;
  static inline RetType apply( const RetType& a, const RetType& b )
    {return ( a * b );}
};

template< class T >
struct Div {
  typedef T RetType;
  static inline RetType apply( const RetType& a, const RetType& b )
    {return ( a / b );}
};


///////////////////////
// boolean operators //
///////////////////////
template< class T >
struct And;

template<>
struct And<bool> {
  typedef bool RetType;
  static inline RetType apply( bool a, bool b )
    {return ( a && b );}
};

template< class T >
struct Or;

template<>
struct Or<bool> {
  typedef bool RetType;
  static inline RetType apply( bool a, bool b )
    {return ( a || b );}
};


//////////////////////////
// Comparison operators //
//////////////////////////
template< class T >
struct Less {
  typedef bool RetType;
  static inline RetType apply( const T& a, const T& b )
    {return ( a < b );}
};

template< class T >
struct LessEqual {
  typedef bool RetType;
  static inline RetType apply( const T& a, const T& b )
    {return ( a <= b );}
};

template< class T >
struct More {
  typedef bool RetType;
  static inline RetType apply( const T& a, const T& b )
    {return ( a > b );}
};

template< class T >
struct MoreEqual {
  typedef bool RetType;
  static inline RetType apply( const T& a, const T& b ) 
    {return ( a >= b );}
};

template< class T >
struct Equal {
  typedef bool RetType;
  static inline RetType apply( const T& a, const T& b )
    {return ( a == b );}
};



template< class A, class Operation, class B >
class BinOp {
  public:
    typedef typename Operation::RetType RetType;
    typedef typename A::ArgType         ArgType;

    BinOp( const A& A, const B& B ):mA(A), mB(B)
      { };

    RetType operator() (const ArgType& x) const
      {return Operation::apply( mA(x), mB(x) );}

  private:
    A mA;
    B mB;
};

/////////////////////////////////
// conv traits for basic types //
/////////////////////////////////
template< class A, class B >
struct BasicConvTrait;

template<>
struct BasicConvTrait< double, double >
  {typedef double ResultType;};

template<>
struct BasicConvTrait< double, int >
  {typedef double ResultType;};

template<>
struct BasicConvTrait< int, double >
  {typedef double ResultType;};

template<>
struct BasicConvTrait< double, float >
  {typedef double ResultType;};

template<>
struct BasicConvTrait< float, double >
  {typedef double ResultType;};

template<>
struct BasicConvTrait< float, int >
  {typedef float ResultType;};

template<>
struct BasicConvTrait< int, float >
  {typedef float ResultType;};
  
template<>
struct BasicConvTrait< int, int >
  {typedef int ResultType;};  
  
template<class A >
struct BasicConvTrait< A, A >
  {typedef A ResultType;};


///////////////////////////////////////////////
// conversion traits for more advanced types //
///////////////////////////////////////////////
template< class A, class B >
struct ConvTrait;

template< class A, class B >
class ConvTrait<Expr<A>, Expr<B> > {
  private:
    typedef typename Expr<A>::RetType LReturn;
    typedef typename Expr<B>::RetType RReturn;
  public:
    typedef typename BasicConvTrait<LReturn,RReturn>::ResultType ResultType;
};

template< class A , class ArgType>
struct ToExprTraits;

template<class ArgType>
struct ToExprTraits<double, ArgType> {
  typedef Expr< ExprLiteral<double, ArgType> >  ToExprType;
};

template<class ArgType>
struct ToExprTraits<int, ArgType>{
  typedef Expr< ExprLiteral<int, ArgType> >  ToExprType;
};


///////////////////////////////
// operators for expressions //
///////////////////////////////
template< template<class T> class Op, class A, class B >
struct operator_trait;

template< template<class T> class Op, class A, class B>
class operator_trait< Expr<A>, Op, Expr<B> > {
  private:
    typedef typename ConvTraits<  Expr<A>, Expr<B> >::ResultType  ResultType;
  public:
    typedef BinOp< Expr<A>, Op< ResultType >, Expr<B> >           ReturnBaseType;
    typedef Expr< ReturnBaseType >                                ReturnType;
};

template< template<class T> class Op, class A, class B >
class operator_trait< Op, A, Expr<B> > {
  private:
    typedef typename Expr<B>::ArgType                              ArgType;
    typedef typename ToExprTrait<A, ArgType>::ToExprType           ToExprType;
    typedef typename ConvTraits ToExprType, Expr<B> >::ResultType  ResultType;
  public:
    typedef ToExprType                                             LToExpr;
    typedef BinOp< ToExprType, Op<ResultType>, Expr<B> >           ReturnBaseType;
    typedef Expr< ReturnBaseType >                                 ReturnType;
};

template< template<class T> class Op, class A, class B >
class operator_trait< Op, Expr<A>, B > {
  private:
    typedef typename Expr<A>::ArgType                              ArgType;
    typedef typename ToExprTrait<B, ArgType>::ToExprType           ToExprType;
    typedef typename ConvTraits< Expr<A>, ToExprType >::ResultType ResultType;
  public:
    typedef ToExprType                                             RToExpr;
    typedef BinOp< Expr<A>, Op<ResultType>, ToExprType >           ReturnBaseType;
    typedef Expr< ReturnBaseType >                                 ReturnType;
};



//////////////////////////
// Expression templates //
//////////////////////////

// operator+ 
template< class A, class B >
typename operator_trait< Expr<A>, Add, Expr<B> >::ReturnType
operator+( const TExpr<A>& A, const TExpr<B>& B ) {
  typedef typename operator_trait< Expr<A>, Add, TExpr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, Add, TExpr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, B) );
}

template< class A, class B >
typename operator_trait< A, Add , Expr<B> >::ReturnType
operator+( const A& A, const Expr<B>& B ) {
  typedef typename operator_trait< A, Add, Expr<B> >::LToExpr        LToExpr;
  typedef typename operator_trait< A, Add, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< A, Add, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(LToExpr(A), B) );
}

template< class A, class B >
typename operator_trait< Expr<A>, Add, B >::ReturnType
operator+( const Expr<A>& A, const B& B ) {
  typedef typename operator_trait< Expr<A>, Add, B >::RToExpr        RToExpr;
  typedef typename operator_trait< Expr<A>, Add, B >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, Add, B >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, RToExpr(B)) );
}


// operator* 
template< class A, class B >
typename operator_trait< Expr<A>, Mul, Expr<B> >::ReturnType
operator*( const Expr<A>& A, const Expr<B>& B ) {
  typedef typename operator_trait< Expr<A>, Mul, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, Mul, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, B) );
}

template< class A, class B >
typename operator_trait< A, Mul, Expr<B> >::ReturnType
operator*( const A& A, const Expr<B>& B ) {
  typedef typename operator_trait< A, Mul, Expr<B> >::LToExpr        LToExpr;
  typedef typename operator_trait< A, Mul, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< A, Mul, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(LToExpr(A), B) );
}

template< class A, class B >
typename operator_trait< Expr<A>, Mul, B  >::ReturnType
operator*( const Expr<A>& A, const B& B ){
  typedef typename operator_trait< Expr<A>, Mul, B >::RToExpr        RToExpr;
  typedef typename operator_trait< Expr<A>, Mul, B >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, Mul, B >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, RToExpr(B)) );
}


// operator/ 
template< class A, class B >
typename operator_trait< Expr<A>, Div, Expr<B> >::ReturnType
operator/( const Expr<A>& A, const Expr<B>& B ) {
  typedef typename operator_trait< Expr<A>, Div, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, Div, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, B) );
}

template< class A, class B >
typename operator_trait< A , Div, Expr<B> >::ReturnType
operator/( const A& A, const Expr<B>& B ) {
  typedef typename operator_trait< A, Div, Expr<B> >::LToExpr        LToExpr;
  typedef typename operator_trait< A, Div, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< A, Div, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(LToExpr(A), B) );
}

template< class A, class B >
typename operator_trait< Expr<A>, Div, B >::ReturnType
operator/( const Expr<A>& A, const B& B ) {
  typedef typename operator_trait< Expr<A>, Div, B >::RToExpr        RToExpr;
  typedef typename operator_trait< Expr<A>, Div, B >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, Div, B >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, RToExpr(B)) );
}


// operator&& 
template< class A, class B >
typename operator_trait< Expr<A>, And, Expr<B> >::ReturnType
operator&&( const Expr<A>& A, const Expr<B>& B )
{
  typedef typename operator_trait< Expr<A>, And, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, And, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, B) );
}


template< class A, class B >
typename operator_trait< A, And, Expr<B> >::ReturnType
operator&&( const A& A, const Expr<B>& B ) {
  typedef typename operator_trait< A, And, Expr<B> >::LToExpr        LToExpr;
  typedef typename operator_trait< A, And, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< A, And, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(LToExpr(A), B) );
}

template< class A, class B >
typename operator_trait< Expr<A>, And, B >::ReturnType
operator&&( const Expr<A>& A, const B& B ) {
  typedef typename operator_trait< Expr<A>, And, B >::RToExpr        RToExpr;
  typedef typename operator_trait< Expr<A>, And, B >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, And, B >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, RToExpr(B)) );
}


// operator|| 
template< class A, class B >
typename operator_trait< Expr<A>, Or, Expr<B> >::ReturnType
operator||( const Expr<A>& A, const Expr<B>& B ) {
  typedef typename operator_trait< Expr<A>, Or, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, Or, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, B) );
}

template< class A, class B >
typename operator_trait< A, Or, Expr<B> >::ReturnType
operator||( const A& A, const Expr<B>& B ) {
  typedef typename operator_trait< A, Or, Expr<B> >::LToExpr        LToExpr;
  typedef typename operator_trait< A, Or, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< A, Or, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(LToExpr(A), B) );
}

template< class A, class B >
typename operator_trait< Expr<A>, Or, B  >::ReturnType
operator||( const Expr<A>& A, const B& B ) {
  typedef typename operator_trait< Expr<A>, Or, B >::RToExpr        RToExpr;
  typedef typename operator_trait< Expr<A>, Or, B >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, Or, B >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, RToExpr(B)) );
}


// operator< 
template< class A, class B >
typename operator_trait< Expr<A>, Less, Expr<B> >::ReturnType
operator<( const Expr<A>& A, const Expr<B>& B ){
  typedef typename operator_trait< Expr<A>, Less, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, Less, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, B) );
}

template< class A, class B >
typename operator_trait< A , Less, TExpr<B> >::ReturnType
operator<( const A& A, const Expr<B>& B ){
  typedef typename operator_trait< A, Less, TExpr<B> >::LToExpr        LToExpr;
  typedef typename operator_trait< A, Less, TExpr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< A, Less, TExpr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(LToExpr(A), B) );
}

template< class A, class B >
typename operator_trait< Expr<A>, Less, B  >::ReturnType
operator<( const Expr<A>& A, const B& B ){
  typedef typename operator_trait< TLt, TExpr<A>, B >::RToExpr        RToExpr;
  typedef typename operator_trait< TLt, TExpr<A>, B >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< TLt, TExpr<A>, B >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, RToExpr(B)) );
}


// operator<= 
template< class A, class B >
typename operator_trait< Expr<A>, LessEqual, Expr<B> >::ReturnType
operator<=( const Expr<A>& A, const Expr<B>& B ) {
  typedef typename operator_trait< Expr<A>, LessEqual, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, LessEqual, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, B) );
}

template< class A, class B >
typename operator_trait< A, LessEqual, Expr<B> >::ReturnType
operator<=( const A& A, const Expr<B>& B ) {
  /// \todo: fixme avoid this by automatic conversion?
  typedef typename operator_trait< A, LessEqual, Expr<B> >::LToExpr        LToExpr;
  typedef typename operator_trait< A, LessEqual, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< A, LessEqual,Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(LToExpr(A), B) );
}

template< class A, class B >
typename operator_trait< Expr<A>, LessEqual, B >::ReturnType
operator<=( const Expr<A>& A, const B& B ) {
  typedef typename operator_trait< Expr<A>, LessEqual, B >::RToExpr        RToExpr;
  typedef typename operator_trait< Expr<A>, LessEqual, B >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, LessEqual, B >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, RToExpr(B)) );
}


// operator> 
template< class A, class B >
typename operator_trait< Expr<A>, More, Expr<B> >::ReturnType
operator>( const Expr<A>& A, const Expr<B>& B ) {
  typedef typename operator_trait< Expr<A>, More, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, More, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, B) );
}

template< class A, class B >
typename operator_trait< A, More, Expr<B> >::ReturnType
operator>( const A& A, const Expr<B>& B ) {
  typedef typename operator_trait< A, More, Expr<B> >::LToExpr        LToExpr;
  typedef typename operator_trait< A, More, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< A, More, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(LToExpr(A), B) );
}

template< class A, class B >
typename operator_trait< Expr<A>, More, B  >::ReturnType
operator>( const Expr<A>& A, const B& B ) {
  typedef typename operator_trait< Expr<A>, More, B >::RToExpr        RToExpr;
  typedef typename operator_trait< Expr<A>, More, B >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, More, B >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, RToExpr(B)) );
}


// operator>= 
template< class A, class B >
typename operator_trait< Expr<A>, MoreEqual, TExpr<B> >::ReturnType
operator>=( const Expr<A>& A, const Expr<B>& B ) {
  typedef typename operator_trait< Expr<A>, MoreEqual, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, MoreEqual, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, B) );
}

template< class A, class B >
typename operator_trait< A , MoreEqual, TExpr<B> >::ReturnType
operator>=( const A& A, const Expr<B>& B ) {
  typedef typename operator_trait< A, MoreEqual, Expr<B> >::LToExpr        LToExpr;
  typedef typename operator_trait< A, MoreEqual, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< A, MoreEqual, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(LToExpr(A), B) );
}

template< class A, class B >
typename operator_trait< Expr<A>, MoreEqual, B  >::ReturnType
operator>=( const Expr<A>& A, const B& B ) {
  typedef typename operator_trait< Expr<A>, MoreEqual, B >::RToExpr        RToExpr;
  typedef typename operator_trait< Expr<A>, MoreEqual, B >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, MoreEqual, B >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, RToExpr(B)) );
}



//------------------------- operator== --------------------------------------------------
template< class A, class B >
typename operator_trait< Expr<A>, Equal, Expr<B> >::ReturnType
operator==( const Expr<A>& A, const Expr<B>& B ) {
  typedef typename operator_trait< Expr<A>, Equal, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, Equal, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, B) );
}

template< class A, class B >
typename operator_trait< A , Equal, Expr<B> >::ReturnType
operator==( const A& A, const Expr<B>& B ) {
  typedef typename operator_trait< A, Equal, Expr<B> >::LToExpr        LToExpr;
  typedef typename operator_trait< A, Equal, Expr<B> >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< A, Equal, Expr<B> >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(LToExpr(A), B) );
}

template< class A, class B >
typename operator_trait< Expr<A>, Equal, B  >::ReturnType
operator==( const Expr<A>& A, const B& B ) {
  typedef typename operator_trait< Expr<A>, Equal, B >::RToExpr        RToExpr;
  typedef typename operator_trait< Expr<A>, Equal, B >::ReturnBaseType ReturnBaseType;
  typedef typename operator_trait< Expr<A>, Equal, B >::ReturnType     ReturnType;
  return ReturnType( ReturnBaseType(A, RToExpr(B)) );
}


}// namespace

#endif 
