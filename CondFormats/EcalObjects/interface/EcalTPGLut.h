#ifndef EcalTPGLut_h
#define EcalTPGLut_h

class EcalTPGLut 
{
 public:

  EcalTPGLut() ;
  EcalTPGLut(const EcalTPGLut &) ;
  ~EcalTPGLut() ;

  EcalTPGLut & operator=(const EcalTPGLut &) ;

  const unsigned int * getLut() const ;
  void setLut(const unsigned int * lut) ;

 private:
  unsigned int lut_[1024] ;

};

#endif
