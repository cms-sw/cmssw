#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/ConnectionPool.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
//
#include <fstream>
#include <iomanip>
// externals
#include "CoralCommon/CommandLine.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"

int main (int argc, char** argv)
{
  std::vector<coral::Option> secondaryOptions;
  //
  coral::Option csPar("conn_string");
  csPar.flag = "-c";
  csPar.helpEntry = "the database connection string";
  csPar.type = coral::Option::STRING;
  secondaryOptions.push_back(csPar);
  //
  coral::Option contPar("container");
  contPar.flag = "-cn";
  contPar.helpEntry = "the selected container name";
  contPar.type = coral::Option::STRING;
  secondaryOptions.push_back(contPar);
  //
  coral::Option mvPar("mapping_version");
  mvPar.flag = "-mv";
  mvPar.helpEntry ="the mapping version";
  mvPar.type = coral::Option::STRING;
  secondaryOptions.push_back(mvPar);
  //
  coral::Option cvPar("class_versions");
  cvPar.flag = "-cv";
  cvPar.helpEntry ="specify the class versions";
  cvPar.type = coral::Option::BOOLEAN;
  secondaryOptions.push_back(cvPar);
  //
  coral::Option outPar("output_file");
  outPar.flag = "-f";
  outPar.helpEntry = "the database host name";
  outPar.type = coral::Option::STRING;
  secondaryOptions.push_back(outPar);
  //
  coral::Option authPar("authentication_path");
  authPar.flag = "-a";
  authPar.helpEntry = "the authentication path";
  authPar.type = coral::Option::STRING;
  secondaryOptions.push_back(authPar);
  //
  coral::Option classPar("type_name");
  classPar.flag = "-t";
  classPar.helpEntry = "the container type name";
  classPar.type = coral::Option::STRING;
  secondaryOptions.push_back(classPar);
  //
  coral::Option dictPar("dictionary");
  dictPar.flag = "-D";
  dictPar.helpEntry = "the list of dictionary libraries";
  dictPar.type = coral::Option::STRING;
  secondaryOptions.push_back(dictPar);
  //
  coral::Option debugPar("debug");
  debugPar.flag = "-debug";
  debugPar.helpEntry ="print the debug messages";
  debugPar.type = coral::Option::BOOLEAN;
  secondaryOptions.push_back(debugPar);
  //
  std::vector<coral::Command> mainSet;
  //
  coral::Command listCont("list_containers");
  listCont.flag = "-list";
  listCont.helpEntry = "listing the available containers";
  listCont.type = coral::Option::BOOLEAN;
  listCont.exclusive = true;
  listCont.addOption(csPar.name);
  listCont.addOption(authPar.name);
  listCont.addOption(debugPar.name);
  mainSet.push_back(listCont);
  //
  coral::Command createCont("create");
  createCont.flag = "-create";
  createCont.helpEntry = "create a database or a container";
  createCont.type = coral::Option::BOOLEAN;
  createCont.exclusive = true;
  createCont.addOption(csPar.name);
  createCont.addOption(contPar.name);
  createCont.addOption(classPar.name);
  createCont.addOption(dictPar.name);
  createCont.addOption(authPar.name);
  createCont.addOption(debugPar.name);
  mainSet.push_back(createCont);
  //
  coral::Command eraseCont("erase");
  eraseCont.flag = "-erase";
  eraseCont.helpEntry = "erase a database or a container";
  eraseCont.type = coral::Option::BOOLEAN;
  eraseCont.exclusive = true;
  eraseCont.addOption(csPar.name);
  eraseCont.addOption(contPar.name);
  eraseCont.addOption(authPar.name);
  eraseCont.addOption(debugPar.name);
  mainSet.push_back(eraseCont);
  //
  coral::Command listMapp("list_mappings");
  listMapp.flag = "-lm";
  listMapp.helpEntry = "listing the available mapping versions";
  listMapp.type = coral::Option::BOOLEAN;
  listMapp.exclusive = true;
  listMapp.addOption(csPar.name);
  listMapp.addOption(contPar.name);
  listMapp.addOption(cvPar.name);
  listMapp.addOption(authPar.name);
  listMapp.addOption(debugPar.name);
  mainSet.push_back(listMapp);
  //
  coral::Command dumpMapp("dump_mapping");
  dumpMapp.flag = "-dm";
  dumpMapp.helpEntry = "dump the specified mapping in xml format";
  dumpMapp.type = coral::Option::BOOLEAN;
  dumpMapp.exclusive = true;
  dumpMapp.addOption(csPar.name);
  dumpMapp.addOption(mvPar.name);
  dumpMapp.addOption(outPar.name);
  dumpMapp.addOption(authPar.name);
  dumpMapp.addOption(debugPar.name);
  mainSet.push_back(dumpMapp);
  //
  try{
    std::string connectionString("");
    std::string authenticationPath("");
    std::string containerName("");
    std::string mappingVersion("");
    std::string fileName("");
    std::string className("");
    std::string dictionary("");
    bool withClassVersion = false;
    bool debug = false;
    coral::CommandLine cmd(secondaryOptions,mainSet);
    cmd.parse(argc,argv);
    const std::map<std::string,std::string>& ops = cmd.userOptions();
    if(cmd.userCommand()==coral::CommandLine::helpOption().name || ops.size()==0){
      cmd.help(std::cout);
      return 0;
    } else {
      std::map<std::string,std::string>::const_iterator iO=ops.find(coral::CommandLine::helpOption().name);
      if(iO!=ops.end()){
        cmd.help(cmd.userCommand(),std::cout);
        return 0;
      } else {
        iO = ops.find(csPar.name);
        if(iO!=ops.end()) {
          connectionString = iO->second;
        } else {
          throw coral::MissingRequiredOptionException(csPar.name);
        }
        iO = ops.find(contPar.name);
        if(iO!=ops.end()) containerName = iO->second;
        iO = ops.find(mvPar.name);
        if(iO!=ops.end()) mappingVersion = iO->second;
        iO = ops.find(authPar.name);
        if(iO!=ops.end()) authenticationPath = iO->second;
        iO = ops.find(outPar.name);
        if(iO!=ops.end()) fileName = iO->second;
        iO = ops.find(classPar.name);
        if(iO!=ops.end()) className = iO->second;
        iO = ops.find(dictPar.name);
        if(iO!=ops.end()) dictionary = iO->second;
        iO = ops.find(cvPar.name);
        if(iO!=ops.end()) withClassVersion = true;
        iO = ops.find(debugPar.name);
        if(iO!=ops.end()) debug = true;

        boost::shared_ptr<ora::ConnectionPool> connection( new ora::ConnectionPool );
        connection->configuration().disablePoolAutomaticCleanUp();
        ora::Database db( connection );
        if( debug ) db.configuration().setMessageVerbosity( coral::Debug );
        db.connect( connectionString );
        ora::ScopedTransaction transaction( db.transaction() );
        transaction.start();

        std::string contTag("container.name");
        std::string classTag("type");
        std::string nobjTag("n.objects");
        std::string mapVerTag("mapping.id");
        std::string classVerTag("class.version");
        std::string space("  ");
        size_t contMax = contTag.length();
        size_t classMax = classTag.length();
        size_t nobjMax = nobjTag.length();
        size_t mapVerMax = mapVerTag.length();
        size_t classVerMax = classVerTag.length();

        if(cmd.userCommand()==listCont.name){
          std::set<std::string> conts = db.containers();

          std::cout << "ORA database in \""<<connectionString<<"\" has "<<conts.size()<<" container(s)."<<std::endl;
          std::cout <<std::endl;
          if( conts.size() ){
            // first find the max lenghts
            for(std::set<std::string>::const_iterator iC = conts.begin(); iC != conts.end(); ++iC ){
              ora::Container cont = db.containerHandle( *iC );
              if(cont.name().length()>contMax ) contMax = cont.name().length();
              if(cont.className().length()>classMax ) classMax = cont.className().length();
            }
            std::cout << std::setiosflags(std::ios_base::left);
            std::cout <<space<<std::setw(contMax)<<contTag;
            std::cout <<space<<std::setw(classMax)<<classTag;
            std::cout <<space<<std::setw(nobjMax)<<nobjTag;
            std::cout <<std::endl;
            
            std::cout <<space<<std::setfill('-');
            std::cout<<std::setw(contMax)<<"";
            std::cout <<space<<std::setw(classMax)<<"";
            std::cout <<space<<std::setw(nobjMax)<<"";
            std::cout <<std::endl;

            std::cout << std::setfill(' ');
            for(std::set<std::string>::const_iterator iC = conts.begin(); iC != conts.end(); ++iC ){
              ora::Container cont = db.containerHandle( *iC );
              std::cout <<space<<std::setw(contMax)<<cont.name();
              std::cout <<space<<std::setw(classMax)<<cont.className();
              std::stringstream ss;
              ss << std::setiosflags(std::ios_base::right);
              ss <<space<<std::setw(nobjMax)<<cont.size();
              std::cout << ss.str();
              std::cout <<std::endl;
            }
          }
          
          transaction.commit();
          return 0;
        }
        if(cmd.userCommand()==createCont.name){
          if( className.empty() ){
            throw coral::MissingRequiredOptionException(classPar.name);
          }
          if( !dictionary.empty() ){
            const boost::filesystem::path dict_path("dictionary");
            edmplugin::SharedLibrary shared( dict_path );
          }
          if( !db.exists() ){
            db.create();
          } else {
            std::set<std::string> conts = db.containers();
            if( conts.find( containerName )!=conts.end() ){
              std::cout << "ERROR: container \"" << containerName << "\" already exists in the database."<<std::endl;
              return -1;
            }
          }
          db.createContainer( className, containerName );
          transaction.commit();
          return 0;
        }
        if(cmd.userCommand()==eraseCont.name){
          if( containerName.empty() ){
            throw coral::MissingRequiredOptionException(contPar.name);
          }
          if( !db.exists() ){
            std::cout << "ERROR: ORA database does not exist."<<std::endl;
            return -1;
          } else {
            std::set<std::string> conts = db.containers();
            if( conts.find( containerName )==conts.end() ){
              std::cout << "ERROR: container \"" << containerName << "\" does not exists in the database."<<std::endl;
              return -1;
            }
            db.dropContainer( containerName );
            transaction.commit();
            return 0;
          }
        }
        if(cmd.userCommand()==listMapp.name){
          if( containerName.empty() ){
            throw coral::MissingRequiredOptionException(contPar.name);
          }
          if( !db.exists() ){
            std::cout << "ERROR: ORA database does not exist."<<std::endl;
            return -1;
          } else {
            ora::DatabaseUtility util = db.utility();
            if(withClassVersion){
              std::map<std::string,std::string> vers = util.listMappings( containerName );

              std::cout << "ORA database in \""<<connectionString<<"\" has "<<vers.size()<<" class version(s) for container \""<<containerName<<"\"."<<std::endl;
              std::cout <<std::endl;
              if( vers.size() ){
                // first find the max lenghts
                for( std::map<std::string,std::string>::const_iterator iM = vers.begin(); iM != vers.end(); ++iM ){
                  if( iM->first.length() > classVerMax ) classVerMax = iM->first.length();
                  if( iM->second.length() > mapVerMax ) mapVerMax = iM->second.length();
                }

                std::cout << std::setiosflags(std::ios_base::left);
                std::cout <<space<<std::setw(classVerMax)<<classVerTag;
                std::cout <<space<<std::setw(mapVerMax)<<mapVerTag;
                std::cout <<std::endl;

                std::cout <<space<<std::setfill('-');
                std::cout<<std::setw(classVerMax)<<"";
                std::cout <<space<<std::setw(mapVerMax)<<"";
                std::cout <<std::endl;

                std::cout << std::setfill(' ');
                for( std::map<std::string,std::string>::const_iterator iM = vers.begin(); iM != vers.end(); ++iM ){
                  std::cout <<space<<std::setw(classVerMax)<<iM->first;
                  std::cout <<space<<std::setw(mapVerMax)<<iM->second;
                  std::cout <<std::endl;
                }
              }              
            } else {
              std::set<std::string> vers = util.listMappingVersions( containerName );

              std::cout << "ORA database in \""<<connectionString<<"\" has "<<vers.size()<<" mapping version(s) for container \""<<containerName<<"\"."<<std::endl;
              std::cout <<std::endl;
              if( vers.size() ){
                // first find the max lenghts
                for( std::set<std::string>::const_iterator iM = vers.begin(); iM != vers.end(); ++iM ){
                  if( iM->length() > mapVerMax ) mapVerMax = iM->length();
                }

                std::cout << std::setiosflags(std::ios_base::left);
                std::cout <<space<<std::setw(mapVerMax)<<mapVerTag;
                std::cout <<std::endl;

                std::cout <<space<<std::setfill('-');
                std::cout <<std::setw(mapVerMax)<<"";
                std::cout <<std::endl;

                std::cout << std::setfill(' ');
                for( std::set<std::string>::const_iterator iM = vers.begin(); iM != vers.end(); ++iM ){
                  std::cout <<space<<std::setw(mapVerMax)<<*iM;
                  std::cout <<std::endl;
                }
              }
            }
            transaction.commit();
            return 0;
          }
        }
        if(cmd.userCommand()==dumpMapp.name){
          if( mappingVersion.empty() ){
            throw coral::MissingRequiredOptionException(mvPar.name);
          }
          if( !db.exists() ){
            std::cout << "ERROR: ORA database does not exist."<<std::endl;
            return -1;
          } else {
            ora::DatabaseUtility util = db.utility();
            std::auto_ptr<std::fstream> file;
            std::ostream* outputStream = &std::cout;
            if( !fileName.empty() ){
              file.reset(new std::fstream);
              file->open( fileName.c_str(),std::fstream::out );
              outputStream = file.get();
            }
            bool dump = util.dumpMapping( mappingVersion, *outputStream );
            if(!dump){
              std::cout << "Mapping with id=\""<<mappingVersion<<"\" has not been found in the database."<<std::endl;
            }
            if( !fileName.empty() ){
              if( dump ) std::cout << "Mapping with id=\""<<mappingVersion<<"\" dumped in file =\""<<fileName<<"\""<<std::endl;
              file->close();
            }
            transaction.commit();
            return 0;
          }
        }
      }
    }

  } catch (const std::exception& e){
    std::cout << "ERROR: " << e.what() << std::endl;
    return -1;
  } catch (...){
    std::cout << "UNEXPECTED FAILURE." << std::endl;
    return -1;
  }
}

