#include "CondCore/RegressionTest/interface/TestFunct.h"

int main(int argc, char **argv)
{
	//reading arguments
	opterr = 0;
	char *wvalue = NULL;
	char *rvalue = NULL;
	char *dvalue = NULL;
	char *svalue = NULL;
	char *Avalue = NULL;
	char *Cvalue = NULL;
	bool cflag =0;
	bool Dflag =0;
	bool Rflag =0;
	int index;
	int c;

	opterr = 0;

	while ((c = getopt (argc, argv, "cDRw:r:d:s:A:C:")) != -1)
		switch (c)
		{
			case 'r':
				rvalue = optarg;
			break;
			case 'R':
				Rflag = 1;
			break;
			case 'w':
				wvalue = optarg;
			case 's':
				svalue = optarg;
			break;
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
				std::cout<<"usage : testCompat [arguments]"<<std::endl;
				std::cout<<"-c creates new TEST_SEED and metadata tables "<<std::endl;
				std::cout<<"-d [mappingName] drops item "<<std::endl;
				std::cout<<"-D drops all items "<<std::endl;
				std::cout<<"-r [mappingName] reads item "<<std::endl;
				std::cout<<"-R reads all items "<<std::endl;
				std::cout<<"-w [mappingName] -s [seed] writes item"<<std::endl;
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
		std::string authenv(std::string("CORAL_AUTH_PATH=")+Avalue);
		::putenv(const_cast<char*>(authenv.c_str()));
		std::string connStr(Cvalue);
		
		edmplugin::PluginManager::Config config;
		edmplugin::PluginManager::configure(edmplugin::standard::config());
		cond::DbConnection conn;// = new cond::DbConnection;
		//conn.configure( cond::CmsDefaults );
		conn.configuration() = cond::DbConnectionConfiguration::defaultConfigurations()[cond::CmsDefaults];
		conn.configuration().setAuthenticationPath(Avalue);
		conn.configure();
		TestFunct tc;
		tc.s = conn.createSession();
		tc.s.open( connStr );

		if(rvalue != NULL)
		{
			std::cout<<"Reading item, mappingName :"<<rvalue<<std::endl;
			if(tc.Read(rvalue) == 1)
			{
				std::cout<<"failed"<<std::endl;
				return 1;
			}
		}
		if(Rflag == 1)
			if(tc.ReadAll() == 1)
			{
				std::cout<<"failed"<<std::endl;
				return 1;
			}
		if(cflag == 1)
		{
			std::cout<<"Creating DB"<<std::endl;
			if (tc.CreateMetaTable() == 1)
			{
				std::cout<<"failed"<<std::endl;
				return 1;
			}
		}
		if(wvalue != NULL)
		{
			if(svalue !=NULL)
			{
				if(tc.Write(wvalue, atoi(svalue)) == 1)
				{
				std::cout<<"failed"<<std::endl;
				::sleep(1);
				return 1;
				}	
				::sleep(1);
			}
			else
			{
				std::cout<<"no seed provided, type -? for help"<<std::endl;
				return 1;
			}
		}
		for (index = optind; index < argc; index++)
			printf ("Non-option argument %s\n", argv[index]);
		if (Dflag == 1)
		{
			if(tc.DropTables(connStr) == 1)
			{
				std::cout<<"failed"<<std::endl;
				return 1;
			}
			else
				std::cout<<"Table Dropped"<<std::endl;
		}
		else if (dvalue != NULL)
		{
			if(tc.DropItem(dvalue) == 1)
			{
				std::cout<<"failed"<<std::endl;
				return 1;
			}
			else
				std::cout<<"Item with mappingName "<<dvalue<<"Dropped"<<std::endl;
		}
		else
		{
			tc.s.close();
		}
		return 0;
	}
	else
	{
		std::cout<<"Database connection parameters are missing"<<std::endl;
		std::cout<<"-A [auth path] -C [connection string]"<<std::endl;
		return 1;
	}
}
