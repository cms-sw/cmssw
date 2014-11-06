#ifndef DataFormats_Luminosity_FillSchemeInfo_h
#define DataFormats_Luminosity_FillSchemeInfo_h
 
/** \class FillSchemeInfo
 *
 *
 * FillSchemeInfo holds information about the filling scheme
 * in the present run
 *
 * \author Josh Bendavid
 *
 ************************************************************/

class FillSchemeInfo {
 
  public:
    FillSchemeInfo() {}
    FillSchemeInfo(int bunchSpacing) : bunchSpacing_(bunchSpacing) {}
    ~FillSchemeInfo() {}
    
    int bunchSpacing() const { return bunchSpacing_; }
    
  private:
    int bunchSpacing_;
  
};

#endif