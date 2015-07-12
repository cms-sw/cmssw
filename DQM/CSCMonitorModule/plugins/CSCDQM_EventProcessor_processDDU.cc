/*
 * =====================================================================================
 *
 *       Filename:  EventProcessor.cc
 *
 *    Description:  EventProcessor Object
 *
 *        Version:  1.0
 *        Created:  10/03/2008 10:47:11 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "CSCDQM_EventProcessor.h"

namespace cscdqm
{

/**
 * @brief  Process DDU output and fill MOs
 * @param  dduData DDU object to process
 */
void EventProcessor::processDDU(const CSCDDUEventData& dduData, const CSCDCCExaminer& binChecker)
{

  CSCDDUHeader dduHeader  = dduData.header();
  CSCDDUTrailer dduTrailer = dduData.trailer();
  if (!dduTrailer.check())
    {
      /**  LOG4CPLUS_WARN(logger_,eTag << "Skipped because of DDU Trailer check failed."); */
      return;
    }

  int dduID = dduHeader.source_id();

  if ( (dduID >= FEDNumbering::MINCSCDDUFEDID) && (dduID <= FEDNumbering::MAXCSCDDUFEDID) )   /// New CSC readout without DCCs. CMS CSC DDU ID range 830-869
    {
      // dduID -= (FEDNumbering::MINCSCDDUFEDID - 1); /// TODO: Can require DDU-RUI remapping for actual system
      dduID = cscdqm::Utility::getRUIfromDDUId(dduHeader.source_id());
      if (dduID < 0)
        {
          LOG_WARN <<  "DDU source ID (" << dduHeader.source_id() << ") is out of valid range. Remapping to DDU ID 1.";
          dduID = 1;
        }
    }
  else
    {

      /**  Only 8bits are significant; format of DDU id is Dxx */
      dduID = dduID & 0xFF;
    }


  MonitorObject* mo = 0;

  if (getEMUHisto(h::EMU_ALL_DDUS_IN_READOUT, mo))
    {
      mo->Fill(dduID);
    }

  std::string dduTag = DDUHistoDef::getPath(dduID);

  uint32_t dduEvtSize = dduData.sizeInWords()*2;

  // if (dduEvtSize > 48)
  {

    /**  DDU word counter */
    int trl_word_count = 0;
    trl_word_count = dduTrailer.wordcount();

    if (getDDUHisto(h::DDU_BUFFER_SIZE, dduID, mo)) mo->Fill(dduEvtSize);

    if (getDDUHisto(h::DDU_WORD_COUNT, dduID, mo)) mo->Fill(trl_word_count );

    /** LOG4CPLUS_DEBUG(logger_,dduTag << " Trailer Word (64 bits) Count = " << std::dec << trl_word_count); */

    if (trl_word_count > 0)
      {
        if (getEMUHisto(h::EMU_ALL_DDUS_EVENT_SIZE, mo))
          {
            mo->Fill(dduID, log10((double)trl_word_count));
          }
      }

    if (getEMUHisto(h::EMU_ALL_DDUS_AVERAGE_EVENT_SIZE, mo))
      {
        mo->Fill(dduID, trl_word_count);
      }

  }

  fCloseL1As = dduTrailer.reserved() & 0x1; // Get status if Close L1As bit
  /**  if (fCloseL1As) LOG4CPLUS_DEBUG(logger_,eTag << " Close L1As bit is set"); */

  /**  DDU Header bunch crossing number (BXN) */
  BXN = dduHeader.bxnum();
  /**  LOG4CPLUS_WARN(logger_,dduTag << " DDU Header BXN Number = " << std::dec << BXN); */
  if (getEMUHisto(h::EMU_DDU_BXN, mo)) mo->Fill(BXN);
  if (getDDUHisto(h::DDU_BXN, dduID, mo)) mo->Fill(BXN);

  /**  L1A number from DDU Header */
  int L1ANumber_previous_event = L1ANumbers[dduID];
  L1ANumbers[dduID] = (int)(dduHeader.lvl1num());
  L1ANumber = L1ANumbers[dduID];
  /** LOG4CPLUS_DEBUG(logger_,dduTag << " Header L1A Number = " << std::dec << L1ANumber); */
  int L1A_inc = L1ANumber - L1ANumber_previous_event;

  /** Handle 24-bit L1A roll-over maximum value case **/
  if ( L1A_inc < 0 ) L1A_inc = 0xFFFFFF + L1A_inc;

  // if (!fFirstEvent) {
  if (fNotFirstEvent[dduID])
    {
      if (getDDUHisto(h::DDU_L1A_INCREMENT, dduID, mo)) mo->Fill(L1A_inc);
      if (getEMUHisto(h::EMU_ALL_DDUS_L1A_INCREMENT, mo))
        {
          if      (L1A_inc > 100000)
            {
              L1A_inc = 19;
            }
          else if (L1A_inc > 30000)
            {
              L1A_inc = 18;
            }
          else if (L1A_inc > 10000)
            {
              L1A_inc = 17;
            }
          else if (L1A_inc > 3000)
            {
              L1A_inc = 16;
            }
          else if (L1A_inc > 1000)
            {
              L1A_inc = 15;
            }
          else if (L1A_inc > 300)
            {
              L1A_inc = 14;
            }
          else if (L1A_inc > 100)
            {
              L1A_inc = 13;
            }
          else if (L1A_inc > 30)
            {
              L1A_inc = 12;
            }
          else if (L1A_inc > 10)
            {
              L1A_inc = 11;
            }
          mo->Fill(dduID, L1A_inc);
        }
    }

  /**  ==     Occupancy and number of DMB (CSC) with Data available (DAV) in header of particular DDU */
  int dmb_dav_header      = 0;
  int dmb_dav_header_cnt  = 0;

  int ddu_connected_inputs= 0;
  int ddu_connected_inputs_cnt = 0;

  int csc_error_state     = 0;
  int csc_warning_state   = 0;

  /**   ==    Number of active DMB (CSC) in header of particular DDU */
  int dmb_active_header   = 0;

  dmb_dav_header       = dduHeader.dmb_dav();
  dmb_active_header    = (int)(dduHeader.ncsc() & 0xF);
  csc_error_state      = dduTrailer.dmb_full()  & 0x7FFF; // Only 15 inputs for DDU
  csc_warning_state    = dduTrailer.dmb_warn()  & 0x7FFF; // Only 15 inputs for DDU
  ddu_connected_inputs = dduHeader.live_cscs();

  /** LOG4CPLUS_DEBUG(logger_,dduTag << " Header DMB DAV = 0x" << std::hex << dmb_dav_header); */
  /** LOG4CPLUS_DEBUG(logger_,dduTag << " Header Number of Active DMB = " << std::dec << dmb_active_header); */

  double freq = 0;
  for (int i = 0;
       i < 15;
       ++i)
    {
      if ((dmb_dav_header >> i) & 0x1)
        {
          dmb_dav_header_cnt++;
          if (getDDUHisto(h::DDU_DMB_DAV_HEADER_OCCUPANCY_RATE, dduID, mo))
            {
              mo->Fill(i + 1);
              freq = (100.0 * mo->GetBinContent(i + 1)) / config->getNEvents();
              if (getDDUHisto(h::DDU_DMB_DAV_HEADER_OCCUPANCY, dduID, mo))
                mo->SetBinContent(i+1,freq);
            }
          if (getEMUHisto(h::EMU_ALL_DDUS_INPUTS_WITH_DATA, mo))
            {
              mo->Fill(dduID, i);
            }
        }

      if ( (ddu_connected_inputs >> i) & 0x1 )
        {
          ddu_connected_inputs_cnt++;
          if (getDDUHisto(h::DDU_DMB_CONNECTED_INPUTS_RATE, dduID, mo))
            {
              mo->Fill(i + 1);
              freq = (100.0 * mo->GetBinContent(i + 1)) / config->getNEvents();
              if (getDDUHisto(h::DDU_DMB_CONNECTED_INPUTS, dduID, mo))
                mo->SetBinContent(i + 1, freq);
            }
          if (getEMUHisto(h::EMU_ALL_DDUS_LIVE_INPUTS, mo))
            {
              mo->Fill(dduID, i);
            }
        }

      if ( (csc_error_state >> i) & 0x1 )
        {
          if (getDDUHisto(h::DDU_CSC_ERRORS_RATE, dduID, mo))
            {
              mo->Fill(i + 1);
              freq = (100.0 * mo->GetBinContent(i + 1)) / config->getNEvents();
              if (getDDUHisto(h::DDU_CSC_ERRORS, dduID, mo))
                mo->SetBinContent(i + 1, freq);
            }
          if (getEMUHisto(h::EMU_ALL_DDUS_INPUTS_ERRORS, mo))
            {
              mo->Fill(dduID, i + 2);
            }
        }

      if ((csc_warning_state >> i) & 0x1 )
        {
          if (getDDUHisto(h::DDU_CSC_WARNINGS_RATE, dduID, mo))
            {
              mo->Fill(i + 1);
              freq = (100.0 * mo->GetBinContent(i + 1)) / config->getNEvents();
              if (getDDUHisto(h::DDU_CSC_WARNINGS, dduID, mo)) mo->SetBinContent(i + 1, freq);
            }
          if (getEMUHisto(h::EMU_ALL_DDUS_INPUTS_WARNINGS, mo))
            {
              mo->Fill(dduID, i + 2);
            }
        }
    }

  if (getEMUHisto(h::EMU_ALL_DDUS_AVERAGE_LIVE_INPUTS, mo))
    {
      mo->Fill(dduID, ddu_connected_inputs_cnt);
    }

  // if (dduEvtSize > 48)
  {
    if (getEMUHisto(h::EMU_ALL_DDUS_AVERAGE_INPUTS_WITH_DATA, mo))
      {
        mo->Fill(dduID, dmb_dav_header_cnt);
      }
  }

  if (getEMUHisto(h::EMU_ALL_DDUS_INPUTS_ERRORS, mo))
    {
      if (csc_error_state > 0)
        {
          mo->Fill(dduID, 1); // Any Input
        }
      else
        {
          mo->Fill(dduID, 0); // No errors
        }
    }

  if (getEMUHisto(h::EMU_ALL_DDUS_INPUTS_WARNINGS, mo))
    {
      if (csc_warning_state > 0)
        {
          mo->Fill(dduID, 1); // Any Input
        }
      else
        {
          mo->Fill(dduID, 0); // No errors
        }
    }

  if (getDDUHisto(h::DDU_DMB_DAV_HEADER_OCCUPANCY, dduID, mo)) mo->SetEntries(config->getNEvents());
  if (getDDUHisto(h::DDU_DMB_CONNECTED_INPUTS, dduID, mo)) mo->SetEntries(config->getNEvents());
  if (getDDUHisto(h::DDU_CSC_ERRORS, dduID, mo)) mo->SetEntries(config->getNEvents());
  if (getDDUHisto(h::DDU_CSC_WARNINGS, dduID, mo)) mo->SetEntries(config->getNEvents());
  if (getDDUHisto(h::DDU_DMB_ACTIVE_HEADER_COUNT, dduID, mo)) mo->Fill(dmb_active_header);
  if (getDDUHisto(h::DDU_DMB_DAV_HEADER_COUNT_VS_DMB_ACTIVE_HEADER_COUNT, dduID, mo))
    mo->Fill(dmb_active_header, dmb_dav_header_cnt);

  /**  Check binary Error status at DDU Trailer */
  uint32_t trl_errorstat = dduTrailer.errorstat();
  if (dmb_dav_header_cnt == 0) trl_errorstat &= ~0x20000000; // Ignore No Good DMB CRC bit of no DMB is present
  /**  LOG4CPLUS_DEBUG(logger_,dduTag << " Trailer Error Status = 0x" << std::hex << trl_errorstat); */
  for (int i = 0; i < 32; i++)
    {
      if ((trl_errorstat >> i) & 0x1)
        {
          if (getDDUHisto(h::DDU_TRAILER_ERRORSTAT_RATE, dduID, mo))
            {
              mo->Fill(i);
              double freq = (100.0 * mo->GetBinContent(i + 1)) / config->getNEvents();
              if (getDDUHisto(h::DDU_TRAILER_ERRORSTAT_FREQUENCY, dduID, mo))
                mo->SetBinContent(i+1, freq);
            }
          if (getDDUHisto(h::DDU_TRAILER_ERRORSTAT_TABLE, dduID, mo))
            mo->Fill(0.,i);
        }
    }
  if (getEMUHisto(h::EMU_ALL_DDUS_TRAILER_ERRORS, mo))
    {
      if (trl_errorstat)
        {
          mo->Fill(dduID, 1); // Any Error
          for (int i = 0; i < 32; i++)
            {
              if ((trl_errorstat >> i) & 0x1)
                {
                  mo->Fill(dduID, i + 2);
                }
            }
        }
      else
        {
          mo->Fill(dduID, 0); // No Errors
        }
    }

  if (getDDUHisto(h::DDU_TRAILER_ERRORSTAT_TABLE, dduID, mo)) mo->SetEntries(config->getNEvents());
  if (getDDUHisto(h::DDU_TRAILER_ERRORSTAT_FREQUENCY, dduID, mo)) mo->SetEntries(config->getNEvents());

  ///** Check DDU Output Path status in DDU Header
  uint32_t ddu_output_path_status = dduHeader.output_path_status();
  if (getEMUHisto(h::EMU_ALL_DDUS_OUTPUT_PATH_STATUS, mo))
    {
      if (ddu_output_path_status)
        {
          mo->Fill(dduID,1); // Any Error
          for (int i=0; i<16; i++)
            {
              if ((ddu_output_path_status>>i) & 0x1)
                {
                  mo->Fill(dduID,i+2); // Fill Summary Histo
                }
            }
        }
      else
        {
          mo->Fill(dduID,0); // No Errors
        }
    }


  uint32_t nCSCs = 0;

  /**  Unpack all found CSC */
  if (config->getPROCESS_CSC())
    {

      std::vector<CSCEventData> chamberDatas;
      chamberDatas.clear();
      chamberDatas = dduData.cscData();

      nCSCs = chamberDatas.size();

      for (uint32_t i = 0; i < nCSCs; i++)
        {
          processCSC(chamberDatas[i], dduID, binChecker);
        }

    }

  if (getDDUHisto(h::DDU_DMB_UNPACKED_VS_DAV, dduID, mo)) mo->Fill(dmb_active_header, nCSCs);

  // fFirstEvent = false;

  /** First event per DDU **/
  fNotFirstEvent[dduID] = true;

}

}
