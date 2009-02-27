// $Id: EEClient.h,v 1.15 2008/06/25 15:08:19 dellaric Exp $

/*!
  \file EEClient.h
  \brief Ecal Barrel Monitor Client mom class
  \author B. Gobbo
  \version $Revision: 1.15 $
  \date $Date: 2008/06/25 15:08:19 $
*/


#ifndef EEClient_H
#define EEClient_H

#include <string>

class EcalCondDBInterface;
class DQMStore;
class RunIOV;
class MonRunIOV;

class EEClient {

 public:

  /*! \fn virtual void analyze(void)
    \brief analyze method
  */
  virtual void analyze(void)      = 0;

  /*! \fn virtual void beginJob(DQMStore* dqmStore)
    \brief begin of job method
  */
  virtual void beginJob(DQMStore* dqmStore)     = 0;

  /*! \fn virtual void endJob(void)
    \brief end of job method
  */
  virtual void endJob(void)       = 0;

  /*! \fn virtual void beginRun(void)
    \brief begin of run method
  */
  virtual void beginRun(void)     = 0;

  /*! \fn virtual void endRun(void)
    \brief end of run method
  */
  virtual void endRun(void)       = 0;

  /*! \fn virtual void setup(void)
    \brief setup method
  */
  virtual void setup(void)        = 0;

  /*! \fn virtual void cleanup(void)
    \brief clean up method
  */
  virtual void cleanup(void)      = 0;

  /*! \fn virtual void softReset(bool flag)
    \brief soft reset method
    \param flag on/off switch
  */
  virtual void softReset(bool flag)    = 0;

  /*! \fn virtual bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag);
    \brief Write data to DataBase
    \param econn DB interface
    \param moniov IOV interface
    \param status good or bad
    \param flag run with softReset or not
  */
  virtual bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) = 0;

  /*! \fn virtual int getEvtPerJob( void );
    \brief Returns the total number of processed events
  */
  virtual int getEvtPerJob( void ) = 0;

  /*! \fn virtual int getEvtPerRun( void );
    \brief Returns the number of processed events in this Run
  */
  virtual int getEvtPerRun( void ) = 0;

  virtual ~EEClient(void) {}

};

#endif // EEClient_H

