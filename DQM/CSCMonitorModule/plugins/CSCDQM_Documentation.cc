/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Documentation.h
 *
 *    Description:  CSCDQM Framework documentation file 
 *
 *        Version:  1.0
 *        Created:  02/19/2009 03:25:23 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

/*! \mainpage CSCDQM Framework Guide
 * 
 * \section intro_sec Introduction
 *
 * CSCDQM Framework provides common services for both Local and Global CSC DQM.
 * It includes analysis module, caching, etc. And should be extended in the
 * nearest future.
 *
 * The rationale in creating this framework/library is being sick in constantly 
 * changing Local and Global DQM code and being not efficient in both places.
 * And having no spare time for hobbies. After starting to program this
 * framework found myself even more busy.
 *
 * Murphys Law: Everything done to improve it only makes it worse. 
 *
 * \section start_sec Quick Start Guide
 *
 * Below are some steps that are necessary to go throught while trying to run
 * the library.
 *
 * \subsection step1 Step 1: Implement cscdqm::MonitorObject
 *
 * One should implement/extend cscdqm::MonitorObject interface which is the
 * main (and only) Monitoring Object that goes into the library and is beinf
 * manipulated uppon. Example:
 *
 * \code
 *
 *  #include "CSCDQM_MonitorObject.h"
 *
 *  class MyMonitorObject : public cscdqm::MonitorObject {
 *
 *    public:
 *
 *      // Implement virtual methods 
 *
 *  };
 *
 * \endcode
 *
 * \subsection step2 Step 2: Implement cscdqm::MonitorObjectProvider
 *
 * cscdqm::MonitorObjectProvider is the object which receives cscdqm::HistoBookRequest
 * to get cscdqm::MonitorObject. Note that particular Monitor Object is being
 * requested only once. After that received pointer to MonitorObject will be
 * held in framework cache efficiently. Example:
 *
 * \code
 *
 * #include "CSCDQM_MonitorObjectProvider.h"
 *
 * class MyMonitorObjectProvider : public cscdqm::MonitorObjectProvider {
 *   
 *   public:
 *
 *     CSCMonitorObject* bookMonitorObject(const cscdqm::HistoBookRequest& req) {
 *
 *        // Your code
 *
 *     }
 *
 *   // Other methods, etc.
 *
 * };
 *
 * \endcode
 *
 * \subsection step3 Step 3: Rock & Roll
 *
 * In your code create cscdqm::Configuration object, supply whatever parameters
 * you need (or load XML configuration file, or edm::ParameterSet), create your 
 * cscdqm::MonitorObjectProvider and then create cscdqm::Dispatcher object. Thats it.
 * Now you can supply events to Dispatcher by calling appropriate method. Note
 * that event processing methods differ in Local and Global DQM! You can call
 * a methof to update fractional and efficiency histograms as well. Example:
 *
 * \code
 *
 * class MyApplication {
 * 
 *   private:
 *
 *     MyMonitorObjectProvider provider;
 *     cscdqm::Configuration   config;
 *     cscdqm::Dispatcher      dispatcher;
 *
 *   public:
 *
 *     MyApplication() : dispatcher(&config, &provider) {
 *       
 *       // do whatever with config object, i.e. 
 *       // config.setPARAMETER(x);
 *
 *       dispatcher.init();
 *     }
 *
 *     ~MyApplication() {
 *       dispatcher.updateFractionAndEfficiencyHistos();
 *     } 
 *
 *     void processEvent(const char* data, const int32_t* dataSize, const uint32_t errorStat, const int32_t nodeNumber) {
 *       dispatcher.processEvent(data, dataSize, errorStat, nodeNumber);
 *     }
 *
 * };
 *
 * \endcode
 *
 */

