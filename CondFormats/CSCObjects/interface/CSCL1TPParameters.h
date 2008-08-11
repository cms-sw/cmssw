#ifndef CSCObjects_CSCL1TPParameters_h
#define CSCObjects_CSCL1TPParameters_h

/** \class CSCL1TPParameters
 *  \author Slava Valuev
 *
 * Description: Configuration parameters needed for the Level-1 CSC Trigger
 *              Primitives emulator.  Expected to be stored in and retrieved
 *              from the conditions database.
 */

class CSCL1TPParameters
{
 public:
  CSCL1TPParameters();
  ~CSCL1TPParameters();

  /** returns ALCT fifo_tbins */
  inline unsigned int alctFifoTbins() const   {return m_alct_fifo_tbins;}

  /** returns ALCT fifo_pretrig */
  inline unsigned int alctFifoPretrig() const {return m_alct_fifo_pretrig;}

  /** returns ALCT bx_width */
  inline unsigned int alctBxWidth() const     {return m_alct_bx_width;}

  /** returns ALCT drift_delay */
  inline unsigned int alctDriftDelay() const  {return m_alct_drift_delay;}

  /** returns ALCT nph_thresh */
  inline unsigned int alctNphThresh() const   {return m_alct_nph_thresh;}

  /** returns ALCT nph_pattern */
  inline unsigned int alctNphPattern() const  {return m_alct_nph_pattern;}

  /** returns ALCT trig_mode */
  inline unsigned int alctTrigMode() const    {return m_alct_trig_mode;}

  /** returns ALCT alct_amode */
  inline unsigned int alctAlctAmode() const   {return m_alct_alct_amode;}

  /** returns ALCT l1a_window */
  inline unsigned int alctL1aWindow() const   {return m_alct_l1a_window;}

  /** returns CLCT fifo_tbins */
  inline unsigned int clctFifoTbins() const   {return m_clct_fifo_tbins;}

  /** returns CLCT fifo_pretrig */
  inline unsigned int clctFifoPretrig() const {return m_clct_fifo_pretrig;}

  /** returns CLCT bx_width */
  inline unsigned int clctBxWidth() const     {return m_clct_bx_width;}

  /** returns CLCT drift_delay */
  inline unsigned int clctDriftDelay() const  {return m_clct_drift_delay;}

  /** returns CLCT nph_pattern */
  inline unsigned int clctNphPattern() const  {return m_clct_nph_pattern;}

  /** returns CLCT hs_thresh */
  inline unsigned int clctHsThresh() const    {return m_clct_hs_thresh;}

  /** returns CLCT ds_thresh */
  inline unsigned int clctDsThresh() const    {return m_clct_ds_thresh;}

  /** returns CLCT hit_thresh */
  inline unsigned int clctHitThresh() const   {return m_clct_hit_thresh;}

  /** returns CLCT pid_thresh */
  inline unsigned int clctPidThresh() const   {return m_clct_pid_thresh;}

  /** returns CLCT sep_src */
  inline unsigned int clctSepSrc() const      {return m_clct_sep_src;}

  /** returns CLCT sep_vme */
  inline unsigned int clctSepVme() const      {return m_clct_sep_vme;}

  /** sets ALCT fifo_tbins */
  void setAlctFifoTbins(const unsigned int theValue) {
    m_alct_fifo_tbins = theValue;
  }

  /** sets ALCT fifo_pretrig */
  void setAlctFifoPretrig(const unsigned int theValue) {
    m_alct_fifo_pretrig = theValue;
  }

  /** sets ALCT bx_width */
  void setAlctBxWidth(const unsigned int theValue) {
    m_alct_bx_width = theValue;
  }

  /** sets ALCT drift_delay */
  void setAlctDriftDelay(const unsigned int theValue) {
    m_alct_drift_delay = theValue;
  }

  /** sets ALCT nph_thresh */
  void setAlctNphThresh(const unsigned int theValue) {
    m_alct_nph_thresh = theValue;
  }

  /** sets ALCT nph_pattern */
  void setAlctNphPattern(const unsigned int theValue) {
    m_alct_nph_pattern = theValue;
  }

  /** sets ALCT trig_mode */
  void setAlctTrigMode(const unsigned int theValue) {
    m_alct_trig_mode = theValue;
  }

  /** sets ALCT alct_amode */
  void setAlctAlctAmode(const unsigned int theValue) {
    m_alct_alct_amode = theValue;
  }

  /** sets ALCT l1a_window */
  void setAlctL1aWindow(const unsigned int theValue) {
    m_alct_l1a_window = theValue;
  }

  /** sets CLCT fifo_tbins */
  void setClctFifoTbins(const unsigned int theValue) {
    m_clct_fifo_tbins = theValue;
  }

  /** sets CLCT fifo_pretrig */
  void setClctFifoPretrig(const unsigned int theValue) {
    m_clct_fifo_pretrig = theValue;
  }

  /** sets CLCT bx_width */
  void setClctBxWidth(const unsigned int theValue) {
    m_clct_bx_width = theValue;
  }

  /** sets CLCT drift_delay */
  void setClctDriftDelay(const unsigned int theValue) {
    m_clct_drift_delay = theValue;
  }

  /** sets CLCT nph_pattern */
  void setClctNphPattern(const unsigned int theValue) {
    m_clct_nph_pattern = theValue;
  }

  /** sets CLCT hs_thresh */
  void setClctHsThresh(const unsigned int theValue) {
    m_clct_hs_thresh = theValue;
  }

  /** sets CLCT ds_thresh */
  void setClctDsThresh(const unsigned int theValue) {
    m_clct_ds_thresh = theValue;
  }

  /** sets CLCT hit_thresh */
  void setClctHitThresh(const unsigned int theValue) {
    m_clct_hit_thresh = theValue;
  }

  /** sets CLCT pid_thresh */
  void setClctPidThresh(const unsigned int theValue) {
    m_clct_pid_thresh = theValue;
  }

  /** sets CLCT sep_src */
  void setClctSepSrc(const unsigned int theValue) {
    m_clct_sep_src = theValue;
  }

  /** sets CLCT sep_vme */
  void setClctSepVme(const unsigned int theValue) {
    m_clct_sep_vme = theValue;
  }

 private:
  /** ALCT configuration parameters. */
  unsigned int m_alct_fifo_tbins, m_alct_fifo_pretrig;
  unsigned int m_alct_bx_width,   m_alct_drift_delay;
  unsigned int m_alct_nph_thresh, m_alct_nph_pattern;
  unsigned int m_alct_trig_mode,  m_alct_alct_amode, m_alct_l1a_window;

  /** CLCT configuration parameters. */
  unsigned int m_clct_fifo_tbins, m_clct_fifo_pretrig;
  unsigned int m_clct_bx_width,   m_clct_drift_delay;
  unsigned int m_clct_nph_pattern;
  unsigned int m_clct_hs_thresh,  m_clct_ds_thresh;
  unsigned int m_clct_hit_thresh, m_clct_pid_thresh;  // new TMB-07 parameters
  unsigned int m_clct_sep_src,    m_clct_sep_vme;
};

#endif
