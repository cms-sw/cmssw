//-------------------------------------------------
//
/**  \class L1MuDTTFMasks
 *
 *   DTTF Masks from OMDS
 *
 *
 *   $Date: 2009/05/12 09:16:28 $
 *   $Revision: 1.1 $
 *
 *   J. Troconiz            UAM Madrid
 */
//
//--------------------------------------------------
#ifndef L1MUDTTF_MASKS_H
#define L1MUDTTF_MASKS_H

//---------------
// C++ Headers --
//---------------


//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuDTTFMasks {

  public:

    /// constructor
    L1MuDTTFMasks() {}

    /// destructor
    virtual ~L1MuDTTFMasks() {}

    /// reset parameters to zero
    void reset();

    void set_inrec_chdis_st1(int wh, int sc, const bool val);
    bool get_inrec_chdis_st1(int wh, int sc) const;
    void set_inrec_chdis_st2(int wh, int sc, const bool val);
    bool get_inrec_chdis_st2(int wh, int sc) const;
    void set_inrec_chdis_st3(int wh, int sc, const bool val);
    bool get_inrec_chdis_st3(int wh, int sc) const;
    void set_inrec_chdis_st4(int wh, int sc, const bool val);
    bool get_inrec_chdis_st4(int wh, int sc) const;
    void set_inrec_chdis_csc(int wh, int sc, const bool val);
    bool get_inrec_chdis_csc(int wh, int sc) const;
    void set_etsoc_chdis_st1(int wh, int sc, const bool val);
    bool get_etsoc_chdis_st1(int wh, int sc) const;
    void set_etsoc_chdis_st2(int wh, int sc, const bool val);
    bool get_etsoc_chdis_st2(int wh, int sc) const;
    void set_etsoc_chdis_st3(int wh, int sc, const bool val);
    bool get_etsoc_chdis_st3(int wh, int sc) const;

    void print() const;

  private:

    int check(int wh, int sc) const;

  private:

    bool inrec_chdis_st1[6][12];
    bool inrec_chdis_st2[6][12];
    bool inrec_chdis_st3[6][12];
    bool inrec_chdis_st4[6][12];
    bool inrec_chdis_csc[6][12];
    bool etsoc_chdis_st1[6][12];
    bool etsoc_chdis_st2[6][12];
    bool etsoc_chdis_st3[6][12];

};

#endif
