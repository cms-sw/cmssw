#ifndef dqm_utils_H
#define dqm_utils_H

// $Id: utils.h,v 1.5 2009/12/15 22:56:08 ameyer Exp $

/*!
  \file utils.h
  \version $Revision: 1.5 $
  \date $Date: 2009/12/15 22:56:08 $
*/

class TH2;

namespace dqm
{
  class utils
  {
  public:
    static void reportSummaryMapPalette(TH2* obj);

  private:
    static bool init;
  };
}

#endif // dqm_utils_H
