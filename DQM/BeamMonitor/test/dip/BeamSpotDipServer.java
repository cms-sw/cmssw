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
  public final static String subject = "dip/CMS/Tracker/BeamSpot";
  DipFactory dip;
  DipData message;
  DipPublication publication;
  double x;
  double y;
  double z;
  double dxdz;
  double dydz;
  double err_x;
  double err_y;
  double err_z;
  double err_dxdz;
  double err_dydz;
  double width_x;
  double width_y;
  double sigma_z;
  double err_width_x;
  double err_width_y;
  double err_sigma_z;

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
      System.out.println("Making publication " + subject);
      publication = dip.createDipPublication(subject, this);
      message = dip.createDipData();
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
	x = ( random.nextGaussian() + 3.67 );
	y = ( random.nextGaussian() - 1.23 );
	z = ( random.nextGaussian() + 9.3456 );

	err_x = Math.abs( random.nextGaussian() * 0.10 );
	err_y = Math.abs( random.nextGaussian() * 0.13 );
	err_z = Math.abs( random.nextGaussian() * 0.18 );

	width_x = Math.abs( random.nextGaussian() * 0.98 );
	width_y = Math.abs( random.nextGaussian() * 1.20 );
	sigma_z = Math.abs( random.nextGaussian() * 35.06 );

	err_width_x = Math.abs( random.nextGaussian() * 0.42 );
	err_width_y = Math.abs( random.nextGaussian() * 0.56 );
	err_sigma_z = Math.abs( random.nextGaussian() * 0.89 );

	dxdz = ( random.nextGaussian() - 4.0);
	dydz = ( random.nextGaussian() - 5.0);
	err_dxdz = Math.abs( random.nextGaussian() *0.59);
	err_dydz = Math.abs( random.nextGaussian() *0.33);

	message.insert("x",x);
	message.insert("y",y);
	message.insert("z",z);
	message.insert("err_x",err_x);
	message.insert("err_y",err_y);
	message.insert("err_z",err_z);
	message.insert("dxdz",dxdz);
	message.insert("dydz",dydz);
	message.insert("err_dxdz",err_dxdz);
	message.insert("err_dydz",err_dydz);
	message.insert("width_x",width_x);
	message.insert("width_y",width_y);
	message.insert("sigma_z",sigma_z);
	message.insert("err_width_x",err_width_x);
	message.insert("err_width_y",err_width_y);
	message.insert("err_sigma_z",err_sigma_z);

	publication.send(message,new DipTimestamp());
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
