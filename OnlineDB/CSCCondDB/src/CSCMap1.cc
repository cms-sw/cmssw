#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"

  /**
   * Constructor for cscmap1
   */
  cscmap1::cscmap1 () throw (oracle::occi::SQLException)
  {
    std::string db_user;
    std::string db_pass;
    env = oracle::occi::Environment::createEnvironment (oracle::occi::Environment::DEFAULT);
    char* c_user = ::getenv("CSCMAP_AUTH_USER");
    char* c_pass = ::getenv("CSCMAP_AUTH_PASSWORD");
    db_user=std::string(c_user);
    db_pass=std::string(c_pass);
    con = env->createConnection (db_user,db_pass,"devdb");
     std::cout << "Connection to mapping DB is done." << std::endl;
  }// end of constructor cscmap1 ()

  /**
   * Destructor for cscmap1.
   */
  cscmap1::~cscmap1 () throw (oracle::occi::SQLException)
  {
    env->terminateConnection (con);
    oracle::occi::Environment::terminateEnvironment (env);
  }  // end of ~cscmap1 ()

  void cscmap1::chamber (int chamberid, CSCMapItem::MapItem *item)
  {
    oracle::occi::Statement *stmt = con->createStatement();
    stmt->setSQL("begin cscmap.chamberid_crate(:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16, :17, :18, :19, :20); end;");

    //std::cout<<"Point1"<<std::endl;

    stmt->setInt (1, chamberid);
    stmt->registerOutParam (2, oracle::occi::OCCISTRING, 10);
    stmt->registerOutParam (3, oracle::occi::OCCIINT);
    stmt->registerOutParam (4, oracle::occi::OCCIINT);
    stmt->registerOutParam (5, oracle::occi::OCCIINT);
    stmt->registerOutParam (6, oracle::occi::OCCIINT);
    stmt->registerOutParam (7, oracle::occi::OCCIINT);
    stmt->registerOutParam (8, oracle::occi::OCCIINT);
    stmt->registerOutParam (9, oracle::occi::OCCIINT);
    stmt->registerOutParam (10, oracle::occi::OCCISTRING, 10);
    stmt->registerOutParam (11, oracle::occi::OCCIINT);
    stmt->registerOutParam (12, oracle::occi::OCCIINT);
    stmt->registerOutParam (13, oracle::occi::OCCIINT);
    stmt->registerOutParam (14, oracle::occi::OCCIINT);
    stmt->registerOutParam (15, oracle::occi::OCCIINT);
    stmt->registerOutParam (16, oracle::occi::OCCIINT);
    stmt->registerOutParam (17, oracle::occi::OCCIINT);
    stmt->registerOutParam (18, oracle::occi::OCCIINT);
    stmt->registerOutParam (19, oracle::occi::OCCIINT);
    stmt->registerOutParam (20, oracle::occi::OCCIINT);

    stmt->execute(); //execute procedure

    //std::cout<<"Point2"<<std::endl;

    item->chamberLabel = stmt->getString(10);
    item->chamberId = chamberid;
     int chamber = chamberid/10%100;
     int rest = (chamberid - chamber*10)/1000;
     int ring = rest%10;
     rest = (rest - ring)/10;
     int station = rest%10;
     int endcap = (rest - station)/10;
    item->endcap = endcap;
    item->station = station;
    item->ring = ring;
    item->chamber = chamber;
    item->cscIndex = stmt->getInt(13);
    item->layerIndex = stmt->getInt(14);
    item->stripIndex = stmt->getInt(15);
    item->anodeIndex = stmt->getInt(16);
    item->strips = stmt->getInt(11);
    item->anodes = stmt->getInt(12);
    item->crateLabel = stmt->getString(2);
    item->crateid  = stmt->getInt(3);
    item->sector = stmt->getInt(8);
    item->trig_sector = stmt->getInt(9);
    item->dmb = stmt->getInt(5);
    item->cscid = stmt->getInt(7);

    stmt->setSQL("begin ddumap.chamberid_ddu(:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16, :17, :18, :19); end;");

    stmt->setInt (1, chamberid);
    stmt->registerOutParam (2, oracle::occi::OCCIINT);
    stmt->registerOutParam (3, oracle::occi::OCCIINT);
    stmt->registerOutParam (4, oracle::occi::OCCIINT);
    stmt->registerOutParam (5, oracle::occi::OCCIINT);
    stmt->registerOutParam (6, oracle::occi::OCCIINT);
    stmt->registerOutParam (7, oracle::occi::OCCIINT);
    stmt->registerOutParam (8, oracle::occi::OCCIINT);
    stmt->registerOutParam (9, oracle::occi::OCCIINT);
    stmt->registerOutParam (10, oracle::occi::OCCISTRING, 10);
    stmt->registerOutParam (11, oracle::occi::OCCIINT);
    stmt->registerOutParam (12, oracle::occi::OCCISTRING, 10);
    stmt->registerOutParam (13, oracle::occi::OCCIINT);
    stmt->registerOutParam (14, oracle::occi::OCCISTRING, 10);
    stmt->registerOutParam (15, oracle::occi::OCCIINT);
    stmt->registerOutParam (16, oracle::occi::OCCIINT);
    stmt->registerOutParam (17, oracle::occi::OCCIINT);
    stmt->registerOutParam (18, oracle::occi::OCCIINT);
    stmt->registerOutParam (19, oracle::occi::OCCIINT);

    stmt->execute(); //execute procedure

    item->ddu = stmt->getInt(2);
    item->ddu_input = stmt->getInt(5);
    item->slink = stmt->getInt(7);

    con->terminateStatement (stmt);
    //std::cout<<"End of cscmap1->chamber"<<std::endl;
  } //end of chamber
