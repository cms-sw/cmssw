//
//  File: BeamSpotDipServer.java   (W.Badgett)
//

import cern.dip.*;
import java.lang.Thread;
import java.util.Random;

public class BeamSpotDipServer 
extends Thread
implements Runnable,DipPublicationErrorHandler
{
  public final static String subjectCMS = "dip/CMS/Tracker/BeamSpot";
  public final static String subjectLHC = "dip/CMS/LHC/LuminousRegion";
  DipFactory dip;
  DipData messageCMS;
  DipData messageLHC;
  DipPublication publicationCMS;
  DipPublication publicationLHC;
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

  public void handleException(DipPublication publication, 
			      DipException e)
  {
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
      keepRunning = false;
    }
    try
    {
      while (keepRunning)
      {
	x = (float)( random.nextGaussian() + 3.67 );
	y = (float)( random.nextGaussian() - 1.23 );
	z = (float)( random.nextGaussian() + 9.3456 );
	Centroid[0] = x;
	Centroid[1] = y;
	Centroid[2] = z;

	err_x = (float)Math.abs( random.nextGaussian() * 0.10 );
	err_y = (float)Math.abs( random.nextGaussian() * 0.13 );
	err_z = (float)Math.abs( random.nextGaussian() * 0.18 );

	width_x = (float)Math.abs( random.nextGaussian() * 0.98 );
	width_y = (float)Math.abs( random.nextGaussian() * 1.20 );
	sigma_z = (float)Math.abs( random.nextGaussian() * 35.06 );
	Size[0] = width_x;
	Size[1] = width_y;
	Size[2] = sigma_z;

	err_width_x = (float)Math.abs( random.nextGaussian() * 0.42 );
	err_width_y = (float)Math.abs( random.nextGaussian() * 0.56 );
	err_sigma_z = (float)Math.abs( random.nextGaussian() * 0.89 );

	dxdz = (float)( random.nextGaussian() - 4.0);
	dydz = (float)( random.nextGaussian() - 5.0);
	Tilt[0] = dxdz;
	Tilt[1] = dydz;

	err_dxdz = (float)Math.abs( random.nextGaussian() *0.59);
	err_dydz = (float)Math.abs( random.nextGaussian() *0.33);

	messageCMS.insert("x",x);
	messageCMS.insert("y",y);
	messageCMS.insert("z",z);
	messageCMS.insert("err_x",err_x);
	messageCMS.insert("err_y",err_y);
	messageCMS.insert("err_z",err_z);
	messageCMS.insert("dxdz",dxdz);
	messageCMS.insert("dydz",dydz);
	messageCMS.insert("err_dxdz",err_dxdz);
	messageCMS.insert("err_dydz",err_dydz);
	messageCMS.insert("width_x",width_x);
	messageCMS.insert("width_y",width_y);
	messageCMS.insert("sigma_z",sigma_z);
	messageCMS.insert("err_width_x",err_width_x);
	messageCMS.insert("err_width_y",err_width_y);
	messageCMS.insert("err_sigma_z",err_sigma_z);

	messageLHC.insert("Size",Size);
	messageLHC.insert("Centroid",Centroid);
	messageLHC.insert("Tilt",Tilt);

	DipTimestamp zeit = new DipTimestamp();
	publicationCMS.send(messageCMS, zeit);
	publicationLHC.send(messageLHC, zeit);
	try { Thread.sleep(5000); }
	catch(InterruptedException e)
	{
	  keepRunning = false;
	}
      }
    } 
    catch (DipException e)
    {
      System.out.println("Failed to send data because " + e.getMessage());
      e.printStackTrace();
    }
  }
    
    
  public static void main(String args[])
  {
    BeamSpotDipServer server = new BeamSpotDipServer();
    server.start();
  }
}
