//
//  File: BeamSpotDipServer.java   (W.Badgett, G.Y.Jeng)
//
//  Some changs were made to publish PVs
//  (Sushil S. Chauhan/ UCDavis)

package cms.dip.tracker.beamspot;

import cern.dip.*;
import java.lang.Thread;
import java.io.*;
import java.text.*;
import java.util.Date;
import java.util.BitSet;

public class BeamSpotDipServer
extends Thread
implements Runnable,DipPublicationErrorHandler
{
  // Input parameters
  public static boolean verbose = false;
  public static boolean overwriteQuality = true; //if true, overwrite quality with qualities[0]
  public static String subjectCMS = "dip/CMS/Tracker/BeamSpot";
  public static String subjectLHC = "dip/CMS/LHC/LuminousRegion";
  public static String subjectPV = "dip/CMS/Tracker/PrimaryVertices";
  public static String sourceFile = "/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults.txt";
  public static String sourceFile1 = "/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults_TkStatus.txt";
  public static int[] timeoutLS = {1,2}; //LumiSections

  // Static variables
  public final static String[] qualities = {"Uncertain","Bad","Good"};
  public final static boolean publishStatErrors = true;
  public final static int secPerLS = 23;
  public final static int rad2urad = 1000000;
  public final static int cm2um = 10000;
  public final static int cm2mm = 10;
  public final static int intLS = 1; //For CMS scaler

  // Coordinate transformation from CMS RF to LHC RF (ref. CMS-TK-UR-0059)
  public final static double[] trans = {-0.09,-0.11,-0.12}; //cm
  public final static double[] angles = {-0.00002,-0.00016,-0.00122}; //rad
  public final static double[] rotX = {Math.cos(angles[0]),Math.sin(angles[0]),Math.tan(angles[0])};
  public final static double[] rotY = {Math.cos(angles[1]),Math.sin(angles[1]),Math.tan(angles[1])};
  public final static double[] rotZ = {Math.cos(angles[2]),Math.sin(angles[2]),Math.tan(angles[2])};

  // DIP objects
  DipFactory dip;
  DipData messageCMS;
  DipData messageLHC;
  DipData messagePV;
  DipPublication publicationCMS;
  DipPublication publicationLHC;
  DipPublication publicationPV;
  // Initial values of Beam Spot object
  int runnum = 0;
  String startTime = getDateTime();
  String endTime = getDateTime();
  long startTimeStamp = 0;
  long endTimeStamp = 0;
  String lumiRange = "0 - 0";
  String quality = "Uncertain";
  int type = -1;
  float x = 0;
  float y = 0;
  float z = 0;
  float dxdz = 0;
  float dydz = 0;
  float err_x = 0;
  float err_y = 0;
  float err_z = 0;
  float err_dxdz = 0;
  float err_dydz = 0;
  float width_x = 0;
  float width_y = 0;
  float sigma_z = 0;
  float err_width_x = 0;
  float err_width_y = 0;
  float err_sigma_z = 0;
  //added for PV infor
  int events = 0;
  float meanPV = 0;
  float err_meanPV = 0;
  float rmsPV = 0;
  float err_rmsPV = 0;
  int maxPV = 0;
  int nPV = 0; 
  //---
  float Size[] = new float[3];
  float Centroid[] = new float[3];
  float Tilt[] = new float[2];

  // Others
  boolean keepRunning;
  long lastFitTime = 0;
  long lastModTime = 0;
  BitSet alive = new BitSet(8);
  int idleTime = 0;
  int lsCount = 0;
  int currentLS = 0;

  public void handleException(DipPublication publication,
			      DipException e)
  {
    System.out.println("handleException: " + getDateTime());
    System.out.println("Error handler for " +
		       publication.getTopicName() +
		       " called because " + e.getMessage());
    e.printStackTrace();
  }

  public void run()
  {
    java.util.Date now = new java.util.Date();

    try
    {
      dip = Dip.create("CmsBeamSpot_"+now.getTime());

      System.out.println("Server Started at " + getDateTime());
      System.out.println("Making publication " + subjectCMS);
      publicationCMS = dip.createDipPublication(subjectCMS, this);
      messageCMS = dip.createDipData();

      System.out.println("Making publication " + subjectLHC);
      publicationLHC = dip.createDipPublication(subjectLHC, this);
      messageLHC = dip.createDipData();

      System.out.println("Making publication " + subjectPV);
      publicationPV = dip.createDipPublication(subjectPV, this);
      messagePV = dip.createDipData();

      trueRcd(false); // Starts with all 0.
      publishRcd("UNINITIALIZED","",true,false);
      keepRunning = true;
    }
    catch ( DipException e )
    {
      System.err.println("DipException [start up]: " + getDateTime());
      keepRunning = false;
    }

    quality = qualities[0];

    while (keepRunning)
    {
      try
      {
        File logFile = new File(sourceFile);
	
	if (!logFile.exists()) {
	    if (verbose) System.out.println("Source File: " + sourceFile + " doesn't exist!");
	    polling();
	    continue;
	}
	else {
	  FileReader fr = new FileReader(logFile);
	  BufferedReader br = new BufferedReader(fr);
	  lastModTime = logFile.lastModified();
	  if (lastFitTime == 0)
	      lastFitTime = lastModTime;
	  if (logFile.length() == 0) {
	      if (lastModTime > lastFitTime) {
		  String tmp = tkStatus();
		  System.out.println("New run starts. Run number: " + runnum);
		  if (verbose) System.out.println("Initial lastModTime = " + getDateTime(lastModTime));
	      }
	      lastFitTime = lastModTime;
	  }

	  if (lastModTime > lastFitTime) {
	      if (verbose) {
		  System.out.println("Time of last fit    = " + getDateTime(lastFitTime));
		  System.out.println("Time of current fit = " + getDateTime(lastModTime));
	      }
	      lastFitTime = lastModTime;
	      if (logFile.length() > 0) {
		  if (verbose) System.out.println("Read record from " + sourceFile);
		  if (readRcd(br)) {
		      trueRcd(true);
		      alive.clear();
		      alive.flip(7);
		  }
		  else fakeRcd();
		  if (verbose) System.out.println("Publish new record");
		  lsCount = 0;
		  idleTime = 0;
	      }
	      br.close();
	      fr.close();
	  }
	  else{
	      br.close();
	      fr.close();
	      polling();
	      continue;
	  }
	}
	// Quality of the publish results
	if (overwriteQuality) publishRcd(qualities[0],"Testing",true,true);
	else if (quality == qualities[1]) publishRcd(quality,"No BeamFit or Fit Fails",true,true);
	else publishRcd(quality,"",true,true);

      } catch (IOException e) {
	  System.err.println("IOException [Loop]: " + getDateTime());
	  e.printStackTrace();
      };
    }
  }

  private void polling()
  {
    if (lsCount != 0 && lsCount%60 == 0) {
	System.out.println("Waiting for data..." + getDateTime());
    }
    try { Thread.sleep(1000); }//every sec
    catch(InterruptedException e) {
	System.err.println("InterruptedException [polling]: " + getDateTime());
	e.printStackTrace();
	keepRunning = false;
    }
    lsCount++;
    idleTime++;
    if ((lsCount%(timeoutLS[0]*secPerLS) == 0)
	&& (lsCount%(timeoutLS[1]*secPerLS) != 0)) {//fist time out
	if (!alive.get(1)) alive.flip(1);
	if (!alive.get(2)) {
	    if (!alive.get(7)) fakeRcd();
	    else trueRcd(false);
	    publishRcd("Uncertain","No new data for " + idleTime + " seconds",false,false);
	}
	else {
	    fakeRcd();
	    String warnMsg = "No new data for " + idleTime + "seconds: ";
	    warnMsg += tkStatus();
	    publishRcd("Bad",warnMsg,false,false);
	}
    }
    else if (lsCount%(timeoutLS[1]*secPerLS) == 0) {//second time out
	if (!alive.get(2)) alive.flip(2);
	//if(!alive.get(7))
	fakeRcd();
	//else trueRcd(false);
	String warnMsg = "No new data for " + idleTime + "seconds: ";
	warnMsg += tkStatus();
	publishRcd("Bad",warnMsg,false,false);
    }
  }

  String tkStatus()
  {
    File logFile = new File(sourceFile1);
    if (!logFile.exists() || logFile.length() == 0) {
	return "No CMS Tracker status available. No DAQ/DQM.";
    }
    else {
      int nthLnInRcd = 0;
      String record = new String();
      String outstr = new String();
      try
	{
	  FileReader fr = new FileReader(logFile);
	  BufferedReader br = new BufferedReader(fr);
	  while ((record = br.readLine()) != null) {
	    //System.out.println(record);
	    nthLnInRcd ++;
	    String[] tmp;
	    tmp = record.split("\\s");
	    switch(nthLnInRcd) {
	    case 7:
		if (!tmp[1].contains("Yes"))
		    outstr = "CMS Tracker OFF.";
		else
		    outstr = "CMS not taking data or No beam.";
		break;
	    case 8:
		runnum = new Integer(tmp[1]);
		break;
	    default:
		break;
	    }
	  }
	  br.close();
	  fr.close();
	}
	catch (Exception e) {
	    System.err.println("Exception [tkStatus]: " + getDateTime());
	    e.printStackTrace();
	}
      return outstr;
    }
  }

  private boolean readRcd(BufferedReader file_)
  {
    int nthLnInRcd = 0;
    String record = new String();
    boolean rcdQlty = false;
    try
    {
      while ((record = file_.readLine()) != null) {
	//System.out.println(record);
	nthLnInRcd ++;
	String[] tmp;
	tmp = record.split("\\s");
	switch(nthLnInRcd) {
	case 1:
 	    if (!record.startsWith("Run")){
 		System.out.println("Reading of results text file interrupted. " + getDateTime());
		return false;
 	    }
	    runnum = new Integer(tmp[1]);
	    System.out.println("Run: " + runnum);
	    break;
	case 2:
	    startTime = tmp[1]+" "+tmp[2]+" "+tmp[3];
	    startTimeStamp = new Long(tmp[4]);
	    //System.out.println("Time of begin run: " + startTime);
	    break;
	case 3:
	    endTime = tmp[1]+" "+tmp[2]+" "+tmp[3];
	    endTimeStamp = new Long(tmp[4]);
	    System.out.println("TimeStamp of fit: " + endTimeStamp + " [sec]");
	    System.out.println("Time of fit: " + endTime);
	    break;
	case 4:
	    lumiRange = record.substring(10);
	    System.out.println("LS: " + lumiRange);
	    currentLS = new Integer(tmp[3]);
	    //System.out.println("Current LS: " + currentLS);
	    break;
	case 5:
	    type = new Integer(tmp[1]);
	    if (overwriteQuality) quality = qualities[0];
	    else if (type >= 2)	quality = qualities[2];
	    else quality = qualities[1];
	    break;
	case 6:
	    x = new Float(tmp[1]);
	    System.out.format("X0 in CMS RF   = %13.7f  [cm]%n", x);
	    break;
	case 7:
	    y = new Float(tmp[1]);
	    System.out.format("Y0 in CMS RF   = %13.7f  [cm]%n", y);
	    break;
	case 8:
	    z = new Float(tmp[1]);
	    System.out.format("Z0 in CMS RF   = %13.7f  [cm]%n", z);
	    break;
	case 9:
	    sigma_z = new Float(tmp[1]);
	    System.out.format("Sigma Z        = %11.5f    [cm]%n", sigma_z);
	    break;
	case 10:
	    dxdz = new Float(tmp[1]);
	    //System.out.println("dxdz           = " + dxdz + " [rad]");
	    break;
	case 11:
	    dydz = new Float(tmp[1]);
	    //System.out.println("dydz           = " + dydz + " [rad]");
	    break;
	case 12:
	    width_x = new Float(tmp[1]);
	    System.out.format("Sigma X        = %14.8f [cm]%n", width_x);
	    break;
	case 13:
	    width_y = new Float(tmp[1]);
	    System.out.format("Sigma Y        = %14.8f [cm]%n", width_y);
	    break;
	case 14:
	    err_x = new Float(Math.sqrt(Double.parseDouble(tmp[1])));
	    //System.out.println(err_x);
	    break;
	case 15:
	    err_y = new Float(Math.sqrt(Double.parseDouble(tmp[2])));
	    //System.out.println(err_y);
	    break;
	case 16:
	    err_z = new Float(Math.sqrt(Double.parseDouble(tmp[3])));
	    //System.out.println(err_z);
	    break;
	case 17:
	    err_sigma_z = new Float(Math.sqrt(Double.parseDouble(tmp[4])));
	    //System.out.println(err_sigma_z);
	    break;
	case 18:
	    err_dxdz = new Float(Math.sqrt(Double.parseDouble(tmp[5])));
	    //System.out.println(err_dxdz);
	    break;
	case 19:
	    err_dydz = new Float(Math.sqrt(Double.parseDouble(tmp[6])));
	    //System.out.println(err_dydz);
	    break;
	case 20:
	    err_width_x = new Float(Math.sqrt(Double.parseDouble(tmp[7])));
	    err_width_y = err_width_x;
            break;
        case 21:                      
            System.out.println("EmittanceX");
            break; 
        case 22:                      
            System.out.println("EmittanceY");
            break; 
        case 23:                      
            System.out.println("BetaStar");
            break; 
        case 24:                      
            events = new Integer(tmp[1]);
            //System.out.println(events);
            break; 
        case 25:                      
            meanPV = new Float(tmp[1]);
            //System.out.println(meanPV);
            break; 
        case 26:                      
            err_meanPV = new Float(tmp[1]);
            //System.out.println(err_meanPV);
            break;
        case 27:                      
            rmsPV = new Float(tmp[1]);
            //System.out.println(rmsPV);
            break; 
        case 28:                      
            err_rmsPV = new Float(tmp[1]);
            //System.out.println(err_rmsPV);
            break;
        case 29:                      
            maxPV = new Integer(tmp[1]);
            //System.out.println(maxPV);
            break;
        case 30:                      
            nPV = new Integer(tmp[1]);
            //System.out.println(nPV);
	    rcdQlty = true;
	    if (verbose) System.out.println("End of reading current record");
	    break;

	default:
	    break;
	}
      }
      file_.close();
    }
    catch (IOException e) {
	System.err.println("IOException [readRcd]: " + getDateTime());
	e.printStackTrace();
    }
    return rcdQlty;
  }

  private void CMS2LHCRF_POS(float x, float y, float z)
  {
    if (x != 0) {//Rotation + Translation + Inversion + Scaling
	double tmpx = x; //*rotY[0]*rotZ[0] + y*rotY[0]*rotZ[1] - z*rotY[1] + trans[0];
	Centroid[0] = new Float(tmpx);
	Centroid[0] *= -1.0*cm2um;
    }
    else
	Centroid[0] = x;
    if (y != 0) {// Rotation + Translation + Scaling
	double tmpy = y; //x*(rotX[1]*rotY[1]*rotZ[0] - rotX[0]*rotZ[1]) + y*(rotX[0]*rotZ[0] + rotX[1]*rotY[1]*rotZ[1]) + z*rotX[1]*rotY[0] + trans[1];
	Centroid[1] = new Float(tmpy);
	Centroid[1] *= cm2um;
    }
    else
	Centroid[1] = y;
    if (z != 0) {//Rotation + Translation + Inversion + Scaling
	double tmpz = z; //x*(rotX[0]*rotY[1]*rotZ[0] + rotX[1]*rotZ[1]) + y*(rotX[0]*rotY[1]*rotZ[1] - rotX[1]*rotZ[0]) + z*rotX[0]*rotY[0] + trans[2];
	Centroid[2] = new Float(tmpz);
	Centroid[2] *= -1.0*cm2mm;
    }
    else
	Centroid[2] = z;
  }

  private void trueRcd(boolean verbose_)
  {
   try
   {
     // CMS to LHC RF
     CMS2LHCRF_POS(x,y,z);

     Tilt[0] = dxdz*rad2urad;
     Tilt[1] = (dydz != 0 ? (dydz*-1*rad2urad) : 0);

     Size[0] = width_x*cm2um;
     Size[1] = width_y*cm2um;
     Size[2] = sigma_z*cm2mm;

     if (verbose_) {
	 System.out.format( "X0 in LHC RF   = %11.5f    [microns]%n", Centroid[0]);
	 System.out.format( "Y0 in LHC RF   = %11.5f    [microns]%n", Centroid[1]);
	 System.out.format( "Z0 in LHC RF   = %13.7f  [mm]%n", Centroid[2]);
     }

     messageCMS.insert("runnum",runnum);
     messageCMS.insert("startTime",startTime);
     messageCMS.insert("endTime",endTime);
     messageCMS.insert("startTimeStamp",startTimeStamp);
     messageCMS.insert("endTimeStamp",endTimeStamp);
     messageCMS.insert("lumiRange",lumiRange);
     messageCMS.insert("quality",quality);
     messageCMS.insert("type",type); //Unknown=-1, Fake=0, Tracker=2(Good)
     messageCMS.insert("x",x);
     messageCMS.insert("y",y);
     messageCMS.insert("z",z);
     messageCMS.insert("dxdz",dxdz);
     messageCMS.insert("dydz",dydz);
     messageCMS.insert("width_x",width_x);
     messageCMS.insert("width_y",width_y);
     messageCMS.insert("sigma_z",sigma_z);
     if (publishStatErrors) {
	 messageCMS.insert("err_x",err_x);
	 messageCMS.insert("err_y",err_y);
	 messageCMS.insert("err_z",err_z);
	 messageCMS.insert("err_dxdz",err_dxdz);
	 messageCMS.insert("err_dydz",err_dydz);
	 messageCMS.insert("err_width_x",err_width_x);
	 messageCMS.insert("err_width_y",err_width_y);
	 messageCMS.insert("err_sigma_z",err_sigma_z);
     }
     messageLHC.insert("Size",Size);
     messageLHC.insert("Centroid",Centroid);
     messageLHC.insert("Tilt",Tilt);
     //start putting values in DIP for PV
     messagePV.insert("runnum",runnum);
     messagePV.insert("startTime",startTime);
     messagePV.insert("endTime",endTime);
     messagePV.insert("startTimeStamp",startTimeStamp);
     messagePV.insert("endTimeStamp",endTimeStamp);
     messagePV.insert("lumiRange",lumiRange);
     messagePV.insert("events",events);   
     messagePV.insert("meanPV",meanPV);  
     messagePV.insert("err_meanPV",err_meanPV);
     messagePV.insert("rmsPV",rmsPV);      
     messagePV.insert("err_rmsPV",err_rmsPV);
     messagePV.insert("maxPV",maxPV);
     messagePV.insert("nPV",nPV);
   } catch (DipException e){
       System.err.println("DipException [trueRcd]: " + getDateTime());
       System.err.println("Failed to send data because " + e.getMessage());
       e.printStackTrace();
   }
  }

  private void fakeRcd()
  {
   try
   {
     Centroid[0] = 0;
     Centroid[1] = 0;
     Centroid[2] = 0;
     
     Size[0] = 0;
     Size[1] = 0;
     Size[2] = 0;
     
     Tilt[0] = 0;
     Tilt[1] = 0;
     
     messageLHC.insert("Size",Size);
     messageLHC.insert("Centroid",Centroid);
     messageLHC.insert("Tilt",Tilt);
   } catch (DipException e){
       System.err.println("DipException [fakeRcd]: " + getDateTime());
       System.err.println("Failed to send data because " + e.getMessage());
       e.printStackTrace();
   }
  }

  private void publishRcd(String qlty_, String err_, boolean pubCMS_, boolean fitTime_)
  {
   try
   {
     boolean updateCMS_ = pubCMS_ && (currentLS%intLS == 0);
     if (alive.get(7)) {
	 if (updateCMS_) System.out.println("Publish record to CCC and CMS (beam spot scaler...)");
	 else
	     if (!alive.get(1) && !alive.get(2)) System.out.println("Publish record to CCC only");
     }
     DipTimestamp zeit;
     if (fitTime_) {
	 long epoch;
	 epoch = endTimeStamp*1000; //convert to ms
	 System.out.println("epoch = " + epoch + " [ms]");
	 zeit = new DipTimestamp(epoch);
     }
     else zeit = new DipTimestamp();

     if(updateCMS_) publicationCMS.send(messageCMS, zeit);
     publicationLHC.send(messageLHC, zeit);
     publicationPV.send(messagePV, zeit);

     if (qlty_ == qualities[0]) {
	 if (updateCMS_) publicationCMS.setQualityUncertain(err_);
	 publicationLHC.setQualityUncertain(err_);
     }
     else if (qlty_ == qualities[1]) {
	 if (updateCMS_) publicationCMS.setQualityBad(err_);
	 publicationLHC.setQualityBad(err_);
     }
     else if (qlty_ == "UNINITIALIZED") {
 	 if (updateCMS_) publicationCMS.setQualityBad("UNINITIALIZED");
	 publicationLHC.setQualityBad("UNINITIALIZED");
     }
   } catch (DipException e){
       System.err.println("DipException [publishRcd]: " + getDateTime());
       System.err.println("Failed to send data because " + e.getMessage());
       e.printStackTrace();
   }
  }

  private String getDateTime()
  {
    DateFormat dateFormat = new SimpleDateFormat("yyyy.MM.dd HH:mm:ss z");
    Date date = new Date();
    return dateFormat.format(date);
  }

  private String getDateTime(long epoch)
  {
    DateFormat dateFormat = new SimpleDateFormat("yyyy.MM.dd HH:mm:ss z");
    Date date = new Date(epoch);
    return dateFormat.format(date);
  }

  private BeamSpotDipServer(String args[])
  {
    this.verbose = args[0].matches("true");
    this.overwriteQuality = args[1].matches("true");
    this.subjectCMS = args[2];
    this.subjectLHC = args[3];
    this.sourceFile = args[4];
    this.timeoutLS[0] = new Integer(args[5]);
    this.timeoutLS[1] = new Integer(args[6]);
    this.sourceFile1 = args[7];
    this.subjectPV = args[8];
  }

  public static void main(String args[])
  {
    BeamSpotDipServer server = new BeamSpotDipServer(args);
    server.start();
  }
}
