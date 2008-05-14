//-------------------------------------------------
//
/**  \class L1MuDTTFParameters
 *
 *   DTTF Parameters from OMDS
 *
 *
 *   $Date: 2008/04/09 15:34:54 $
 *   $Revision: 1.5 $
 *
 *   J. Troconiz            UAM Madrid
 */
//
//--------------------------------------------------
#ifndef L1MUDTTF_PARAMETERS_H
#define L1MUDTTF_PARAMETERS_H

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


class L1MuDTTFParameters {

  public:

    /// constructor
    L1MuDTTFParameters() {}

    /// destructor
    virtual ~L1MuDTTFParameters() {}

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

    void set_inrec_qual_st1(int wh, int sc, const unsigned short int val);
    unsigned short int get_inrec_qual_st1(int wh, int sc) const;
    void set_inrec_qual_st2(int wh, int sc, const unsigned short int val);
    unsigned short int get_inrec_qual_st2(int wh, int sc) const;
    void set_inrec_qual_st3(int wh, int sc, const unsigned short int val);
    unsigned short int get_inrec_qual_st3(int wh, int sc) const;
    void set_inrec_qual_st4(int wh, int sc, const unsigned short int val);
    unsigned short int get_inrec_qual_st4(int wh, int sc) const;

    void set_soc_stdis_n(int wh, int sc, const unsigned short int val);
    unsigned short int get_soc_stdis_n(int wh, int sc) const;
    void set_soc_stdis_wl(int wh, int sc, const unsigned short int val);
    unsigned short int get_soc_stdis_wl(int wh, int sc) const;
    void set_soc_stdis_wr(int wh, int sc, const unsigned short int val);
    unsigned short int get_soc_stdis_wr(int wh, int sc) const;
    void set_soc_stdis_zl(int wh, int sc, const unsigned short int val);
    unsigned short int get_soc_stdis_zl(int wh, int sc) const;
    void set_soc_stdis_zr(int wh, int sc, const unsigned short int val);
    unsigned short int get_soc_stdis_zr(int wh, int sc) const;

    void set_soc_qcut_st1(int wh, int sc, const unsigned short int val);
    unsigned short int get_soc_qcut_st1(int wh, int sc) const;
    void set_soc_qcut_st2(int wh, int sc, const unsigned short int val);
    unsigned short int get_soc_qcut_st2(int wh, int sc) const;
    void set_soc_qcut_st4(int wh, int sc, const unsigned short int val);
    unsigned short int get_soc_qcut_st4(int wh, int sc) const;
    void set_soc_qual_csc(int wh, int sc, const unsigned short int val);
    unsigned short int get_soc_qual_csc(int wh, int sc) const;

    void set_soc_run_21(int wh, int sc, const bool val);
    bool get_soc_run_21(int wh, int sc) const;
    void set_soc_nbx_del(int wh, int sc, const bool val);
    bool get_soc_nbx_del(int wh, int sc) const;
    void set_soc_csc_etacanc(int wh, int sc, const bool val);
    bool get_soc_csc_etacanc(int wh, int sc) const;
    void set_soc_openlut_extr(int wh, int sc, const bool val);
    bool get_soc_openlut_extr(int wh, int sc) const;

  private:

    int check(int wh, int sc) const;

  private:

    bool inrec_chdis_st1[6][12];
    bool inrec_chdis_st2[6][12];
    bool inrec_chdis_st3[6][12];
    bool inrec_chdis_st4[6][12];

    unsigned short int inrec_qual_st1[6][12];
    unsigned short int inrec_qual_st2[6][12];
    unsigned short int inrec_qual_st3[6][12];
    unsigned short int inrec_qual_st4[6][12];

    unsigned short int soc_stdis_n[6][12];
    unsigned short int soc_stdis_wl[6][12];
    unsigned short int soc_stdis_wr[6][12];
    unsigned short int soc_stdis_zl[6][12];
    unsigned short int soc_stdis_zr[6][12];

    unsigned short int soc_qcut_st1[6][12];
    unsigned short int soc_qcut_st2[6][12];
    unsigned short int soc_qcut_st4[6][12];
    unsigned short int soc_qual_csc[6][12];

    bool soc_run_21[6][12];
    bool soc_nbx_del[6][12];
    bool soc_csc_etacanc[6][12];
    bool soc_openlut_extr[6][12];

};

#endif
