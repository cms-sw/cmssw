//
//  File: BeamSpotDipServer.java   (W.Badgett, G.Y.Jeng)
//

package cms.dip.tracker.beamspot;

import cern.dip.*;
import java.lang.Thread;
import java.util.Random;
import java.io.*;
import java.text.*;
import java.util.Date;
import java.util.BitSet;

public class BeamSpotDipServer
extends Thread
implements Runnable,DipPublicationErrorHandler
{
  public final static boolean appendRcd = false; //if true, appending latest results
  public final static boolean overwriteQuality = true; //if true, change quality to qualities[0]
  public final static boolean publishStatErrors = false;
  public final static String subjectCMS = "dip/CMS/Tracker/BeamSpot";
  public final static String subjectLHC = "dip/CMS/LHC/LuminousRegion";
  public final static String sourceFile = "/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt";
  public final static int secPerLS = 23;
  public final static int rad2urad = 1000000;
  public final static int cm2um = 10000;
  public final static int cm2mm = 10;
  public final static String[] qualities = {"Uncertain","Bad","Good"};
  public final static int[] idleLS = {5,10}; //LumiSections

  DipFactory dip;
  DipData messageCMS;
  DipData messageLHC;
  DipPublication publicationCMS;
  DipPublication publicationLHC;
  int runnum;
  String startTime;
  String endTime;
  String lumiRange;
  String quality;
  int type;
  float x;
  float y;
  float z;
  float dxdz;
  float dydz;
  float err_x;
  float err_y;
  float err_z;
  float err_dxdz;
  float err_dydz;
  float width_x;
  float width_y;
  float sigma_z;
  float err_width_x;
  float err_width_y;
  float err_sigma_z;
  float Size[] = new float[3];
  float Centroid[] = new float[3];
  float Tilt[] = new float[2];

  boolean keepRunning;
  Random random = new Random((long)0xadeadcdf);
  long lastFitTime = 0;
  BitSet alive = new BitSet(8);
  int idleTime = 0;

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

      keepRunning = true;
    }
    catch ( DipException e )
    {
      System.err.println("DipException: " + getDateTime());
      keepRunning = false;
    }

    int lsCount = 0;
    quality = qualities[0];
    try
    {
      File logFile = new File(sourceFile);
      logFile.createNewFile();
      long filePointer = logFile.length();
      RandomAccessFile myFile;

      while (keepRunning)
      {
	try
	{
	  if (logFile.createNewFile()) {
	      String commands = "chmod a+w " + sourceFile;
	      Process proc1 = Runtime.getRuntime().exec(commands);
	  }
	  myFile = new RandomAccessFile(logFile,"r");
	  long fileLength = logFile.length();

	  if (fileLength < filePointer) {
	      System.out.println("New Run Started");
	      filePointer = 0;
	      lsCount = 0;
	      myFile.close();
	      continue;
	  }
	  else if (fileLength == filePointer) {
	      myFile.close();
	      if (lsCount%60 == 0)
		  System.out.println("Waiting for data...");
	      try { Thread.sleep(1000); }//every sec
	      catch(InterruptedException e) {
		  System.err.println("InterruptedException: " + getDateTime());
		  e.printStackTrace();
		  keepRunning = false;
	      }
	      lsCount++;
	      idleTime++;
	      if ((lsCount < idleLS[1]*secPerLS) && (lsCount%(idleLS[0]*secPerLS) == 0)) {//fist time out
		  if (!alive.get(1)) alive.flip(1);
		  if(!alive.get(7)) fakeRcd();
		  else trueRcd();
		  if(!alive.get(2)) publishRcd("Uncertain","No new data for " + idleTime + " seconds",false,false);
		  else publishRcd("Bad","No new data for " + idleTime + " seconds",false,false);
	      }
	      else if (lsCount%(idleLS[1]*secPerLS) == 0) {//second time out
		  if (!alive.get(2)) alive.flip(2);
		  if(!alive.get(7)) fakeRcd();
		  else trueRcd();
		  publishRcd("Bad","No new data for " + idleTime + " seconds",false,false);
		  lsCount = 0;
	      }
	      continue;
	  }
	  else {
	      System.out.println("Read new record from " + sourceFile);
	      if (appendRcd) filePointer = readRcd(myFile,filePointer);
	      else filePointer = readRcd(myFile,0);
	      trueRcd();
	      myFile.close();
	      lsCount = 0;
	      idleTime = 0;
	      alive.clear();
	      alive.flip(7);
	  }
	} catch (Exception e){
	    System.err.println("Exception: " + getDateTime());
	    e.printStackTrace();
	}
	if (overwriteQuality) publishRcd(qualities[0],"Testing",true,true);
	else if (quality == qualities[1]) publishRcd(quality,"BeamFit does not converge",true,true);
	else publishRcd(quality,"",true,true);
      }
    } catch (IOException e) {
	System.err.println("IOException: " + getDateTime());
	e.printStackTrace();
    };
  }

  private long readRcd(RandomAccessFile file_, long filePointer_)
  {
    int nthLnInRcd = 0;
    String record = new String();
    try
    {
      file_.seek(filePointer_);
      while ((record = file_.readLine()) != null) {
	//System.out.println(record);
	nthLnInRcd ++;
	String[] tmp;
	tmp = record.split("\\s");
	switch(nthLnInRcd) {
	case 1:
	    if (!record.startsWith("Run")){
		System.out.println("BeamFitResults text file may be corrupted.");
		System.out.println("Stopping BeamSpot DIP Server!");
		System.exit(0);
	    }
	    runnum = new Integer(tmp[1]);
	    System.out.println("Run: " + runnum);
	    break;
	case 2:
	    startTime = record.substring(15);
	    //System.out.println("Time of begin run: " + startTime);
	    break;
	case 3:
	    endTime = record.substring(13);
	    System.out.println("Time of fit: " + endTime);
	    break;
	case 4:
	    lumiRange = record.substring(10);
	    System.out.println("LS: " + lumiRange);
	    break;
	case 5:
	    type = new Integer(tmp[1]);
	    if (overwriteQuality) quality = qualities[0];
	    else if (type >= 2)	quality = qualities[2];
	    else quality = qualities[1];
	    break;
	case 6:
	    x = new Float(tmp[1]);
	    System.out.println("x0      = " + x + " [cm]");
	    break;
	case 7:
	    y = new Float(tmp[1]);
	    System.out.println("y0      = " + y + " [cm]");
	    break;
	case 8:
	    z = new Float(tmp[1]);
	    System.out.println("z0      = " + z + " [cm]");
	    break;
	case 9:
	    sigma_z = new Float(tmp[1]);
	    System.out.println("sigma_z = " + sigma_z + " [cm]");
	    break;
	case 10:
	    dxdz = new Float(tmp[1]);
	    break;
	case 11:
	    dydz = new Float(tmp[1]);
	    break;
	case 12:
	    width_x = new Float(tmp[1]);
	    break;
	case 13:
	    width_y = new Float(tmp[1]);
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
			
	default:
	    break;
	}
      }
      filePointer_ = file_.getFilePointer();
    }
    catch (IOException e) {
	System.err.println("IOException: " + getDateTime());
	e.printStackTrace();
    }
    return filePointer_;
  }

  private void trueRcd()
  {
   try
   {
     Centroid[0] = x*-1*cm2um;
     Centroid[1] = y*cm2um;
     Centroid[2] = z*-1*cm2mm;
	  
     Size[0] = width_x*cm2um;
     Size[1] = width_y*cm2um;
     Size[2] = sigma_z*cm2mm;
	  
     Tilt[0] = dxdz*rad2urad;
     Tilt[1] = dydz*-1*rad2urad;
	  
     messageCMS.insert("runnum",runnum);
     messageCMS.insert("startTime",startTime);
     messageCMS.insert("endTime",endTime);
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
   } catch (DipException e){
       System.err.println("DipException: " + getDateTime());
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
       System.err.println("DipException: " + getDateTime());
       System.err.println("Failed to send data because " + e.getMessage());
       e.printStackTrace();
   }
  }

  private void publishRcd(String qlty_,String err_, boolean pubCMS_, boolean fitTime_)
  {
   try
   {
     try
     {
      DipTimestamp zeit;
      if (fitTime_) {
	  long epoch = new SimpleDateFormat("yyyy.MM.dd HH:mm:ss zz").parse(endTime).getTime();
	  //System.out.println(epoch);
	  zeit = new DipTimestamp(epoch);
      }
      else zeit = new DipTimestamp();

      if(pubCMS_) publicationCMS.send(messageCMS, zeit);
      publicationLHC.send(messageLHC, zeit);
     } catch (ParseException e) {
	 System.err.println("ParseException: " + getDateTime());
	 System.err.println("Publishing failed due to time parsing because " + e.getMessage());
	 e.printStackTrace();
     }

     if (qlty_ == qualities[0]) {
	  if (pubCMS_) publicationCMS.setQualityUncertain(err_);
	  publicationLHC.setQualityUncertain(err_);
      }
      else if (qlty_ == qualities[1]) {
	  if (pubCMS_) publicationCMS.setQualityBad(err_);
	  publicationLHC.setQualityBad(err_);
      }
   } catch (DipException e){
       System.err.println("DipException: " + getDateTime());
       System.err.println("Failed to send data because " + e.getMessage());
       e.printStackTrace();
   }
  }

  private String getDateTime()
  {
    DateFormat dateFormat = new SimpleDateFormat("yyyy.MM.dd HH:mm:ss");
    Date date = new Date();
    return dateFormat.format(date);
  }

  public static void main(String args[])
  {
    BeamSpotDipServer server = new BeamSpotDipServer();
    server.start();
  }
}
