#include "CondCore/RegressionTest/interface/TestFunct.h"

int main(int argc, char **argv)
{
	//reading arguments
	opterr = 0;
	char *dvalue = NULL;
	char *Avalue = NULL;
	char *Cvalue = NULL;
	bool cflag =0;
	bool Dflag =0;
	int c;

	opterr = 0;

	while ((c = getopt (argc, argv, "cDd:A:C:")) != -1)
		switch (c)
		{
			case 'c':
				cflag = 1;
			break;
			case 'd':
				dvalue = optarg;
			break;
			case 'D':
				Dflag = 1;
			break;
			case 'A':
				Avalue = optarg;
			break;
			case 'C':
				Cvalue = optarg;
			break;
			case '?':
				std::cout<<"usage : setupDB [arguments]"<<std::endl;
				std::cout<<"-c creates new TEST_SEED and metadata tables "<<std::endl;
				std::cout<<"-d [mappingName] drops item "<<std::endl;
				std::cout<<"-D drops all items "<<std::endl;
				std::cout<<"afterwards supply the following arguments :"<<std::endl;
				std::cout<<"-A [auth path] -C [connection string]"<<std::endl;
				return 1;
			default:
				std::cout<<"bad syntax, type -? for help"<<std::endl;
				return 1;
		}
	if(Avalue !=NULL && Cvalue != NULL)
	{
		// std::string user("SSAMAITI");
		// std::string passwd("SimSam123");
		// std::string connStr("oracle://devdb11/ssamaiti");
		//std::string userenv(std::string("CORAL_AUTH_USER=")+uvalue);
		//std::string pwdenv(std::string("CORAL_AUTH_PASSWORD=")+pvalue);
		//::putenv(const_cast<char*>(userenv.c_str()));
		//::putenv(const_cast<char*>(pwdenv.c_str()));
		std::string aut("/afs/cern.ch/user/s/ssamaiti/scratch0/Test/");
		std::string authenv(std::string("CORAL_AUTH_PATH=")+aut);
		::putenv(const_cast<char*>(authenv.c_str()));
		std::string connStr(Cvalue);
		
		edmplugin::PluginManager::Config config;
		edmplugin::PluginManager::configure(edmplugin::standard::config());
		cond::DbConnection conn;// = new cond::DbConnection;
		conn.configuration() = cond::DbConnectionConfiguration::defaultConfigurations()[cond::CmsDefaults];
		conn.configuration().setAuthenticationPath(Avalue);
		conn.configure();
		//conn.configuration().setAuthenticationPath(aut);
		//conn.configure( cond::CmsDefaults );
		TestFunct tc;
		tc.s = conn.createSession();
		tc.s.open( connStr );
		
		if(cflag == 1)
		{
			std::cout<<"Creating DB"<<std::endl;
			if (!tc.CreateMetaTable())
			{
				std::cout<<"failed"<<std::endl;
				return 1;
			}
		}
		if (Dflag == 1)
		{
			if(!tc.DropTables(connStr))
			{
				std::cout<<"failed"<<std::endl;
				return 1;
			}
			else
				std::cout<<"Table Dropped"<<std::endl;
		}
		else if (dvalue != NULL)
		{
			if(!tc.DropItem(dvalue))
			{
				std::cout<<"failed"<<std::endl;
				return 1;
			}
			else
				std::cout<<"Item with mappingName "<<dvalue<<"Dropped"<<std::endl;
		}
		tc.s.close();
		return 0;
	}
	else
	{
		std::cout<<"Database connection parameters are missing"<<std::endl;
		std::cout<<"-A [auth path] -C [connection string]"<<std::endl;
		return 1;
	}
}
