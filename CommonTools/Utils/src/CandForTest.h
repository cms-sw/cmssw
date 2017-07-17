// Only for Unit test
namespace eetest {
  struct CandForTest {

    CandForTest(){}
    CandForTest(float p,float e, float q) : d{p,e,q}{}

    float pt() const  { return d[0]; }  
    float eta() const { return d[1]; }  
    float phi() const { return d[2]; }  

    float d[3];

  };
}
