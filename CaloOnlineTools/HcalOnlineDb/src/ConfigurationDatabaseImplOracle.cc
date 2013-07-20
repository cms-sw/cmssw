// $Id: ConfigurationDatabaseImplOracle.cc,v 1.4 2013/05/23 15:17:36 gartung Exp $

#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImplOracle.hh"

#include "xgi/Utils.h"
#include "toolbox/string.h"

using namespace std;
using namespace oracle::occi;
using namespace hcal;
//
// provides factory method for instantion of ConfigurationDatabaseImplOracle application
//
DECLARE_PLUGGABLE(hcal::ConfigurationDatabaseImpl,ConfigurationDatabaseImplOracle)

ConfigurationDatabaseImplOracle::ConfigurationDatabaseImplOracle()
{

}

bool ConfigurationDatabaseImplOracle::canHandleMethod(const std::string& method) const {
  return (method=="occi");
}

ConfigurationDatabaseImplOracle::~ConfigurationDatabaseImplOracle() {
	disconnect();

}

void ConfigurationDatabaseImplOracle::connect(const std::string& accessor) throw (hcal::exception::ConfigurationDatabaseException) {
  std::map<std::string,std::string> params;
  std::string user, host, method, db, port,password;
  ConfigurationDatabaseImpl::parseAccessor(accessor,method,host,port,user,db,params);
  
  if (method!="occi") XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,std::string("Invalid accessor for Oracle : ")+accessor);
    
  if (params.find("PASSWORD")!=params.end()) password=params["PASSWORD"];
  if (params.find("LHWM_VERSION")!=params.end()) lhwm_version=params["LHWM_VERSION"];

  try {    
     env_ = oracle::occi::Environment::createEnvironment (oracle::occi::Environment::DEFAULT);
     conn_ = env_->createConnection (user, password, db);
   } catch (SQLException& e) {
           XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
   }


  if (env_ == NULL || conn_ == NULL) {
    std::string message("Error connecting on accessor '");
    message+=accessor;
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,message);
  }
}

void ConfigurationDatabaseImplOracle::disconnect() {

   try {
        //terminate environment and connection
        env_->terminateConnection(conn_);
        Environment::terminateEnvironment(env_);
   } catch (SQLException& e) {
           XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
   }


}

//Utility function that cnverts oracle::occi::Clob to std::string
string ConfigurationDatabaseImplOracle::clobToString(const oracle::occi::Clob& _clob){
		oracle::occi::Clob clob = _clob;
                Stream *instream = clob.getStream (1,0);
		unsigned int size = clob.length();
                char *cbuffer = new char[size];
                memset (cbuffer, 0, size);
                instream->readBuffer (cbuffer, size);
                std::string str(cbuffer,size);
		return str;
}
//inline function to convert hex2integer
inline static int cvtChar(int c) {
  if (c>='0' && c<='9') c-='0';
  if ((c>='a' && c<='f')||(c>='A' && c<='F')) c-='A'+10;
  return c&0xF;
}

void ConfigurationDatabaseImplOracle::getLUTChecksums(const std::string& tag, 
		std::map<hcal::ConfigurationDatabase::LUTId, 
		hcal::ConfigurationDatabase::MD5Fingerprint>& checksums) throw (hcal::exception::ConfigurationDatabaseException) {

	if (env_ == NULL || conn_ == NULL) XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Database is not open");
  	checksums.clear();

   try {
        Statement* stmt = conn_->createStatement();

        std::string query = ("SELECT TRIG_PRIM_LOOKUPTBL_DATA_CLOB FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_TRIG_LOOKUP_TABLES");
        query+=" WHERE CRATE=-1";
        //query+=toolbox::toString(" WHERE TAG_NAME='%s' CRATE=-1", tag.c_str());

        //SELECT
        ResultSet *rs = stmt->executeQuery(query.c_str());

        while (rs->next()) {
                oracle::occi::Clob clob = rs->getClob (1);
                std::list<ConfigurationDatabaseStandardXMLParser::Item> items;
                std::string buffer = clobToString(clob);

                m_parser.parseMultiple(buffer,items);
		
                for (std::list<ConfigurationDatabaseStandardXMLParser::Item>::iterator i=items.begin(); i!=items.end(); ++i) {
	
                        hcal::ConfigurationDatabase::FPGASelection ifpga = 
				(hcal::ConfigurationDatabase::FPGASelection)atoi(i->parameters["topbottom"].c_str());
                        int islot = atoi(i->parameters["slot"].c_str());
			hcal::ConfigurationDatabase::LUTType ilut_type = 
				(hcal::ConfigurationDatabase::LUTType)atoi(i->parameters["luttype"].c_str());
			int crate=atoi(i->parameters["crate"].c_str());
                        int islb=atoi(i->parameters["slb"].c_str());
                        int islbch=atoi(i->parameters["slbchan"].c_str());
			hcal::ConfigurationDatabase::LUTId lut_id;
                        lut_id=hcal::ConfigurationDatabase::LUTId(crate, islot, ifpga, islb, islbch, ilut_type);

			hcal::ConfigurationDatabase::MD5Fingerprint csum(16);
			std::string encoded = (std::string)i->items[0];
			for (int i=0; i<16; i++) {
				//converting hex2integer
				csum[i]=cvtChar(encoded[i*2])*16+cvtChar(encoded[i*2+1]);
			}

			checksums[lut_id]=csum;
		}
	}
        //Always terminate statement
        conn_->terminateStatement(stmt);
   } catch (SQLException& e) {
           XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
   }

}

void ConfigurationDatabaseImplOracle::getLUTs(const std::string& tag, int crate, int slot, std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::LUT >& LUTs) throw (hcal::exception::ConfigurationDatabaseException) {
  if (m_lutCache.crate!=crate || m_lutCache.tag!=tag) {
    m_lutCache.clear();
    getLUTs_real(tag,crate,m_lutCache.luts);
    m_lutCache.crate=crate;
    m_lutCache.tag=tag;
  }

  LUTs.clear();
  std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::LUT >::const_iterator i;
  for (i=m_lutCache.luts.begin(); i!=m_lutCache.luts.end(); i++) {
    if (i->first.slot==slot)
      LUTs.insert(*i);
  }
}


void ConfigurationDatabaseImplOracle::getLUTs_real(const std::string& tag, int crate,  
			std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::LUT >& LUTs) 
								throw (hcal::exception::ConfigurationDatabaseException)
{

   try {
	//Lets run the SQl Query
	Statement* stmt = conn_->createStatement();

	std::string query = ("SELECT TRIG_PRIM_LOOKUPTBL_DATA_CLOB FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_TRIG_LOOKUP_TABLES");
	query+=toolbox::toString(" WHERE TAG_NAME='%s' AND CRATE=%d", tag.c_str(), crate);

	//SELECT
        ResultSet *rs = stmt->executeQuery(query.c_str());

  	LUTs.clear();

	while (rs->next()) {
		oracle::occi::Clob clob = rs->getClob (1);
		std::list<ConfigurationDatabaseStandardXMLParser::Item> items;
		std::string buffer = clobToString(clob);
		m_parser.parseMultiple(buffer,items);

      		for (std::list<ConfigurationDatabaseStandardXMLParser::Item>::iterator i=items.begin(); i!=items.end(); ++i) {
			hcal::ConfigurationDatabase::FPGASelection ifpga = 
					(hcal::ConfigurationDatabase::FPGASelection)atoi(i->parameters["TOPBOTTOM"].c_str());
			int islot = atoi(i->parameters["SLOT"].c_str());

			//If this is the desired slot
			//if (islot == slot) {
				hcal::ConfigurationDatabase::LUTType ilut_type = 
						(hcal::ConfigurationDatabase::LUTType)atoi(i->parameters["LUT_TYPE"].c_str());
				hcal::ConfigurationDatabase::LUTId lut_id;
        			if (ilut_type==hcal::ConfigurationDatabase::LinearizerLUT) {
					int ifiber=atoi(i->parameters["FIBER"].c_str());
					int ifiberch=atoi(i->parameters["FIBERCHAN"].c_str());
					lut_id=hcal::ConfigurationDatabase::LUTId(crate, islot, ifpga, ifiber, ifiberch, ilut_type);
				} else {
					int islb=atoi(i->parameters["SLB"].c_str());
					int islbch=atoi(i->parameters["SLBCHAN"].c_str());
					lut_id=hcal::ConfigurationDatabase::LUTId(crate, islot, ifpga, islb, islbch, ilut_type);
				}

				hcal::ConfigurationDatabase::LUT lut;
				lut.reserve(i->items.size());

				int strtol_base=0;
				if (i->encoding=="hex") strtol_base=16;
				else if (i->encoding=="dec") strtol_base=10;

				// convert the data
				for (unsigned int j=0; j<i->items.size(); j++) 
					lut.push_back(strtol(i->items[j].c_str(),0,strtol_base));

				LUTs.insert(make_pair(lut_id, lut));
			//}    
    		}
	}

	//Always terminate statement
	conn_->terminateStatement(stmt);
   } catch (SQLException& e) {
           XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
   }

}

void ConfigurationDatabaseImplOracle::getPatterns(const std::string& tag, int crate, int slot, std::map<hcal::ConfigurationDatabase::PatternId, hcal::ConfigurationDatabase::HTRPattern >& patterns) throw (hcal::exception::ConfigurationDatabaseException) {
  if (m_patternCache.crate!=crate || m_patternCache.tag!=tag) {
    m_patternCache.clear();
    getPatterns_real(tag,crate,m_patternCache.patterns);
    m_patternCache.crate=crate;
    m_patternCache.tag=tag;
  }

  patterns.clear();
  std::map<hcal::ConfigurationDatabase::PatternId, hcal::ConfigurationDatabase::HTRPattern >::const_iterator i;
  for (i=m_patternCache.patterns.begin(); i!=m_patternCache.patterns.end(); i++) {
    if (i->first.slot==slot)
      patterns.insert(*i);
  }
}

void ConfigurationDatabaseImplOracle::getPatterns_real(const std::string& tag, int crate,  
			std::map<hcal::ConfigurationDatabase::PatternId, hcal::ConfigurationDatabase::HTRPattern >& patterns) 
							throw (hcal::exception::ConfigurationDatabaseException) {
   try {
        //Lets run the SQl Query
        Statement* stmt = conn_->createStatement();

        std::string query = ("SELECT HTR_DATA_PATTERNS_DATA_CLOB FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_HTR_DATA_PATTERNS");
        query+=toolbox::toString(" WHERE TAG_NAME='%s' AND CRATE=%d", tag.c_str(), crate);

        //SELECT
        ResultSet *rs = stmt->executeQuery(query.c_str());

        patterns.clear();

        while (rs->next()) {
                oracle::occi::Clob clob = rs->getClob (1);
                std::list<ConfigurationDatabaseStandardXMLParser::Item> items;
                std::string buffer = clobToString(clob);

                m_parser.parseMultiple(buffer,items);

                for (std::list<ConfigurationDatabaseStandardXMLParser::Item>::iterator i=items.begin(); i!=items.end(); ++i) {
                        int islot=atoi(i->parameters["SLOT"].c_str());
                        //If this is the desired slot
                        //if (islot == slot) {
                                hcal::ConfigurationDatabase::FPGASelection ifpga =
					(hcal::ConfigurationDatabase::FPGASelection)atoi(i->parameters["TOPBOTTOM"].c_str());
                                int ifiber=atoi(i->parameters["FIBER"].c_str());

                                hcal::ConfigurationDatabase::PatternId pat_id(crate, islot, ifpga, ifiber);
                                hcal::ConfigurationDatabase::HTRPattern& pat=patterns[pat_id];
                                pat.reserve(i->items.size());

                                int strtol_base=0;
                                if (i->encoding=="hex") strtol_base=16;
                                else if (i->encoding=="dec") strtol_base=10;

                                // convert the data
                                for (unsigned int j=0; j<i->items.size(); j++)
                                        pat.push_back(strtol(i->items[j].c_str(),0,strtol_base));
                        //}
                }
        }
        //Always terminate statement
        conn_->terminateStatement(stmt);
   } catch (SQLException& e) {
           XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
   }


}

void ConfigurationDatabaseImplOracle::getRBXdata(const std::string& tag, const std::string& rbx,
                        hcal::ConfigurationDatabase::RBXdatumType dtype,
                        std::map<ConfigurationDatabase::RBXdatumId, hcal::ConfigurationDatabase::RBXdatum>& RBXdata)
                        throw (hcal::exception::ConfigurationDatabaseException) {

        RBXdata.clear();

        Statement* stmt = conn_->createStatement();
        std::string query;

        //std::string datatypestr;
        switch (dtype) {

                case (ConfigurationDatabase::eRBXpedestal):
                        //datatypestr="PEDESTAL";

                        //Lets run the SQl Query
                        query  = "SELECT MODULE_POSITION, QIE_CARD_POSITION, QIE_ADC_NUMBER, INTEGER_VALUE ";
                        query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_RBX_PEDESTAL_CONFIG ";
                        query += toolbox::toString(" WHERE TAG_NAME='%s' AND NAME_LABEL='%s'", tag.c_str(), rbx.c_str());

                        break;

                case (ConfigurationDatabase::eRBXdelay):
                        //datatypestr="DELAY";

                        query  = "SELECT MODULE_POSITION, QIE_CARD_POSITION, QIE_ADC_NUMBER, INTEGER_VALUE ";
                        query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_RBX_DELAY_CONFIG ";
                        query += toolbox::toString(" WHERE TAG_NAME='%s' AND NAME_LABEL='%s'", tag.c_str(), rbx.c_str());

                        break;

                case (ConfigurationDatabase::eRBXgolCurrent):
                        //datatypestr="GOL";

                        query  = "SELECT MODULE_POSITION, QIE_CARD_POSITION, FIBER_NUMBER, INTEGER_VALUE ";
                        query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_RBX_GOL_CONFIG ";
                        query += toolbox::toString(" WHERE TAG_NAME='%s' AND NAME_LABEL='%s'", tag.c_str(), rbx.c_str());

                        break;
                case (ConfigurationDatabase::eRBXledData):
                        //datatypestr="LED";
                        query  = "SELECT LED_AMPLITUDE, SET_LEDS_IS_CHECKED, BUNCH_NUMBER ";
                        query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_RBX_INITPAR_T02_CONFIG ";
                        query += toolbox::toString(" WHERE TAG_NAME='%s' AND NAME_LABEL='%s'", tag.c_str(), rbx.c_str());
                        break;


                case (ConfigurationDatabase::eRBXbroadcast):
                        //datatypestr="";
                        printf("ConfigurationDatabaseImplMySQL::getRBXdata Can't handle BROADCAST yet\n");
                        return;
                case (ConfigurationDatabase::eRBXttcrxPhase):
                        //datatypestr="PHASE";
                        printf("ConfigurationDatabaseImplMySQL::getRBXdata Can't handle TTCRX PHASE yet\n");
                        return;
                case (ConfigurationDatabase::eRBXqieResetDelay):
                        //datatypestr="";
                        printf("ConfigurationDatabaseImplMySQL::getRBXdata Can't handle QIE RESET DELAY yet\n");
                        return;
                case (ConfigurationDatabase::eRBXccaPatterns):
                        XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Patterns must use getRBXPatterns, not getRBXData");
                        return;
                default:
                      printf("ConfigurationDatabaseImplMySQL::getRBXdata Can't handle dtype=%d yet\n",dtype);
                      return;
        }

   try {
        //SELECT
        ResultSet *rs = stmt->executeQuery(query.c_str());
        while (rs->next()) {

                if (dtype==ConfigurationDatabase::eRBXledData) {
                        //LED_AMPLITUDE, SET_LEDS_IS_CHECKED, BUNCH_NUMBER
                        unsigned int ampl = rs->getInt(1);
                        unsigned int enable = rs->getInt(2);
                        unsigned int bunch = rs->getInt(3);

                        if (enable) enable|=0x1; // global enable if either is on
                        RBXdata.insert(std::pair<ConfigurationDatabase::RBXdatumId,ConfigurationDatabase::RBXdatum>
                                (ConfigurationDatabase::eLEDenable,enable));
                        RBXdata.insert(std::pair<ConfigurationDatabase::RBXdatumId,ConfigurationDatabase::RBXdatum>
                                (ConfigurationDatabase::eLEDamplitude,ampl));
                        RBXdata.insert(std::pair<ConfigurationDatabase::RBXdatumId,ConfigurationDatabase::RBXdatum>
                                (ConfigurationDatabase::eLEDtiming_hb,((bunch&0xFF00)>>8)));
                        RBXdata.insert(std::pair<ConfigurationDatabase::RBXdatumId,ConfigurationDatabase::RBXdatum>
                                (ConfigurationDatabase::eLEDtiming_lb,(bunch&0xFF)));
                } else {
                        //MODULE_POSITION, QIE_CARD_POSITION, QIE_ADC_NUMBER/FIBER_NUMBER, INTEGER_VALUE
                        int rm = rs->getInt(1);
                        int card = rs->getInt(2);
                        int qie_or_gol = rs->getInt(3);
                        unsigned int data = rs->getInt(4);

                        ConfigurationDatabase::RBXdatumId id(rm,card,qie_or_gol,dtype);
                        RBXdata.insert(std::pair<ConfigurationDatabase::RBXdatumId,ConfigurationDatabase::RBXdatum>(id,(unsigned char)(data)));
                }
        }

        //Always terminate statement
        conn_->terminateStatement(stmt);
   } catch (SQLException& e) {
           XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
   }

}

void ConfigurationDatabaseImplOracle::getZSThresholds(const std::string& tag, int crate, int slot,
                std::map<hcal::ConfigurationDatabase::ZSChannelId, int>& thresholds)
                throw (hcal::exception::ConfigurationDatabaseException) {

   try {
        //Lets run the SQl Query
        Statement* stmt = conn_->createStatement();
        //SELECT HTR_FIBER, FIBER_CHANNEL, ZERO_SUPPRESSION, HTR_FPGA
        //FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_ZERO_SUPPRESSION_LHWM
        //WHERE TAG_NAME='Kukartsev test 1' AND CRATE=2 AND HTR_SLOT=2
	//AND LHWM_VERSION='20'

        std::string query = ("SELECT HTR_FIBER, FIBER_CHANNEL, ZERO_SUPPRESSION, HTR_FPGA ");
        query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_ZERO_SUPPRESSION_LHWM ";
        query+=toolbox::toString(" WHERE TAG_NAME='%s' AND CRATE=%d AND HTR_SLOT=%d", tag.c_str(), crate, slot);
        query+=toolbox::toString(" AND LHWM_VERSION='%s'", lhwm_version.c_str());

        //SELECT
        ResultSet *rs = stmt->executeQuery(query.c_str());

        thresholds.clear();

        while (rs->next()) {
                unsigned int fiber = rs->getInt(1);
                unsigned int fc = rs->getInt(2);
                unsigned int zs = rs->getInt(3);
                std::string fpga = rs->getString(4);
                int tb;
                if (fpga=="top") tb = 1;
                else tb = 0;
                std::cout << "crate,slot,tb,fiber,fc:" << crate<<slot<<tb<<fiber<<fc<<std::endl;
                hcal::ConfigurationDatabase::ZSChannelId id(crate,slot,(hcal::ConfigurationDatabase::FPGASelection)tb,fiber,fc);
                thresholds[id] = zs;
        }
        //Always terminate statement
        conn_->terminateStatement(stmt);
   } catch (SQLException& e) {
           XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
   }

}


void ConfigurationDatabaseImplOracle::getHLXMasks(const std::string& tag, int crate, int slot,
                        std::map<hcal::ConfigurationDatabase::FPGAId, hcal::ConfigurationDatabase::HLXMasks>& masks)
                                        throw (hcal::exception::ConfigurationDatabaseException) {
  if (m_hlxMaskCache.crate!=crate || m_hlxMaskCache.tag!=tag) {
    m_hlxMaskCache.clear();
    getHLXMasks_real(tag,crate,m_hlxMaskCache.masks);
    m_hlxMaskCache.crate=crate;
    m_hlxMaskCache.tag=tag;
  } 
  
  masks.clear();
  std::map<ConfigurationDatabase::FPGAId, ConfigurationDatabase::HLXMasks>::const_iterator i;
  for (i=m_hlxMaskCache.masks.begin(); i!=m_hlxMaskCache.masks.end(); i++) {
    if (i->first.slot==slot)
      masks.insert(*i);
  }
} 


void ConfigurationDatabaseImplOracle::getHLXMasks_real(const std::string& tag, int crate,
                std::map<ConfigurationDatabase::FPGAId, ConfigurationDatabase::HLXMasks>& masks)
                throw (hcal::exception::ConfigurationDatabaseException) {
   try {
        //Lets run the SQl Query
        Statement* stmt = conn_->createStatement();
        std::string query = ("SELECT SLOT_NUMBER, FPGA, OCC_MASK, LHC_MASK, SUM_ET_MASK ");
        query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.V_HCAL_HLX_MASKS ";
        query += toolbox::toString(" WHERE TAG_NAME='%s' AND CRATE_NUMBER=%d ", tag.c_str(), crate);

        //SELECT
        ResultSet *rs = stmt->executeQuery(query.c_str());
        masks.clear();
        while (rs->next()) {
                int islot = rs->getInt(1);
                std::string fpga = rs->getString(2);

                int ifpga;
                if (fpga=="top") ifpga = 1;
                else ifpga = 0;

                hcal::ConfigurationDatabase::FPGAId fpga_id;
                fpga_id=hcal::ConfigurationDatabase::FPGAId(crate, islot,
                                (hcal::ConfigurationDatabase::FPGASelectionEnum)ifpga);
                hcal::ConfigurationDatabase::HLXMasks hlxMask;
                hlxMask.occMask = rs->getInt(3);
                hlxMask.lhcMask = rs->getInt(4);
                hlxMask.sumEtMask = rs->getInt(5);

                masks[fpga_id]=hlxMask;
        }
        //Always terminate statement
        conn_->terminateStatement(stmt);
   } catch (SQLException& e) {
           XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
   }
}

// added by Gena Kukartsev
oracle::occi::Connection * ConfigurationDatabaseImplOracle::getConnection( void ){
  return conn_;
}

oracle::occi::Environment * ConfigurationDatabaseImplOracle::getEnvironment( void ){
  return env_;
}

