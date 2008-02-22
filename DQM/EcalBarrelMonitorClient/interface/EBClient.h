// $Id: EBClient.h,v 1.9 2007/03/26 17:35:04 dellaric Exp $

/*!
  \file EBClient.h
  \brief Ecal Barrel Monitor Client mom class
  \author B. Gobbo
  \version $Revision: 1.9 $
  \date $Date: 2007/03/26 17:35:04 $
*/


#ifndef EBClient_H
#define EBClient_H

#include <vector>
#include <string>

class EcalCondDBInterface;
class RunIOV;
class MonRunIOV;

class EBClient {

 public:

  /*! \fn virtual void subscribe(void)
    \brief Subscribe to Monitoring Elements
  */
  virtual void subscribe(void)    = 0;

  /*! \fn virtual void subscribeNew(void)
    \brief Subscribe to Monitoring Elements
  */
  virtual void subscribeNew(void) = 0;

  /*! \fn virtual void unsubscribe(void)
    \brief Unsubscribe to Monitoring Elements
  */
  virtual void unsubscribe(void)  = 0;

  /*! \fn virtual void unsubscribe(void)
    \brief softReset Monitoring Elements
  */
  virtual void softReset(void)  = 0;

  /*! \fn virtual void analyze(void)
    \brief analyze method
  */
  virtual void analyze(void)      = 0;

  /*! \fn virtual void beginJob(MonitorUserInterface* mui)
    \brief Begin of job method
  */
  virtual void beginJob(MonitorUserInterface* mui)     = 0;

  /*! \fn virtual void endJob(void)
    \brief End of Job method
  */
  virtual void endJob(void)       = 0;

  /*! \fn virtual void beginRun(void)
    \brief Begin of Run method
  */
  virtual void beginRun(void)     = 0;

  /*! \fn virtual void endRun(void)
    \brief End of Run method
  */
  virtual void endRun(void)       = 0;

  /*! \fn virtual void setup(void)
    \brief setup method
  */
  virtual void setup(void)        = 0;

  /*! \fn virtual void cleanup(void)
    \brief Clean up method
  */
  virtual void cleanup(void)      = 0;

  /*! \fn virtual void htmlOutput(int run, string htmlDir, string htmlName);
    \brief create HTML page
    \param run run number
    \param htmlDir path to HTML file
    \param htmlName HTML file name

  */
  virtual void htmlOutput(int run, string htmlDir, string htmlName) = 0;

  /*! \fn virtual bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov);
    \brief Write data to DataBase
    \param econn DB interface
    \param moniov IOV interface
  */
  virtual bool writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov) = 0;

  /*! \fn virtual int getEvtPerJob( void );
    \brief Returns the total number of processed events
  */
  virtual int getEvtPerJob( void ) = 0;

  /*! \fn virtual int getEvtPerRun( void );
    \brief Returns the number of processed events in this Run
  */
  virtual int getEvtPerRun( void ) = 0;

  virtual ~EBClient(void) {}

};

#endif // EBClient_H

