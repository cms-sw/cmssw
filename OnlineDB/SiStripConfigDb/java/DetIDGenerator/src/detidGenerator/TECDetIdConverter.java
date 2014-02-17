package detidGenerator;

/**
 * <p>Used to convert the det id to a 32 bits word</p>
 * @author G. Baulieu
 * @version 1.0
**/

/*
  $Date: 2007/03/22 20:46:49 $
  
  $Log: TECDetIdConverter.java,v $
  Revision 1.3  2007/03/22 20:46:49  gbaulieu
  New numbering of Det ID

  Revision 1.2  2006/08/30 15:21:12  gbaulieu
  Add the TOB analyzer

  Revision 1.1  2006/06/28 11:42:24  gbaulieu
  First import of the sources

  Revision 1.5  2006/05/05 15:01:56  baulieu
  Correct the stereo value:
  0 normal mono
  1 stereo
  2 glued mono

  Revision 1.4  2006/03/21 17:04:13  baulieu
  New version of the TEC det_id (no fw/bw modules)

  Revision 1.3  2006/02/22 08:40:21  baulieu
  Invert the DCU_IDs

  Revision 1.2  2006/02/08 15:03:00  baulieu
  Add the convertion to 32 bits for the TOB


*/

public class TECDetIdConverter extends DetIdConverter{

    private int tec;
    private int wheel;
    private int frontBack;
    private int petal;
    private int ring;
    private int modNumber;
    private int stereo;

    /// two bits would be enough, but  we could use the number "0" as a wildcard
    private final short sideStartBit =           18;
    private final short wheelStartBit =          14;
    private final short petal_fw_bwStartBit =    12;
    private final short petalStartBit =          8;
    private final short ringStartBit =           5;
    private final short detStartBit =         2;
    private final short sterStartBit =           0;
    
    /// two bits would be enough, but  we could use the number "0" as a wildcard
    private final short sideMask =          0x3;
    private final short wheelMask =         0xF;
    private final short petal_fw_bwMask =   0x3;
    private final short petalMask =         0xF;
    private final short ringMask =          0x7;
    private final short detMask =        0x7;
    private final short sterMask =          0x3;


    public TECDetIdConverter(int t, int w, int fb, int p, int r, int mn, int s){
	super(1, 6);
	tec = t;
	wheel = w;
	frontBack = fb;
	petal = p;
	ring = r;
	modNumber = mn;
	stereo = s;
    }

    public TECDetIdConverter(String detID) throws Exception{
	super(1,6);
	try{
	    String[] val = detID.split("\\.");
	    if(val.length!=9)
		throw new Exception("The detID has an invalid format");
	    else{
	
		tec = Integer.parseInt(val[2]);
		wheel = Integer.parseInt(val[3]);
		frontBack = Integer.parseInt(val[4]);
		petal = Integer.parseInt(val[5]);
		ring = Integer.parseInt(val[6]);
		modNumber = Integer.parseInt(val[7]);
		stereo = Integer.parseInt(val[8]);
	    }
	}
	catch(NumberFormatException e){
	    throw new Exception("TECDetIdConverter : \n"+e.getMessage());
	}
    }

    public TECDetIdConverter(int detID) throws Exception{
	super(detID);
	tec = getTEC();
	wheel = getWheel();
	frontBack = getFrontBack();
	petal = getPetal();
	ring = getRing();
	modNumber = getModNumber();
	stereo = getStereo();
    }

    public int compact(){
	super.compact();
	id |= (tec&sideMask)<<sideStartBit |
	    (wheel&wheelMask)<<wheelStartBit |
	    (frontBack&petal_fw_bwMask)<<petal_fw_bwStartBit |
	    (petal&petalMask)<<petalStartBit |
	    (ring&ringMask)<<ringStartBit |
	    (modNumber&detMask)<<detStartBit |
	    (stereo&sterMask)<<sterStartBit;
	return id;
    }

    public int getTEC(){
	return (id>>sideStartBit)&sideMask;
    }

    public int getWheel(){
	return (id>>wheelStartBit)&wheelMask;
    }

    public int getFrontBack(){
	return (id>>petal_fw_bwStartBit)&petal_fw_bwMask;
    }
    
    public int getPetal(){
	return (id>>petalStartBit)&petalMask;
    }

    public int getRing(){
	return (id>>ringStartBit)&ringMask;
    }

    public int getModNumber(){
	return (id>>detStartBit)&detMask;
    }

    public int getStereo(){
	return (id>>sterStartBit)&sterMask;
    }

    public String toString(){
	return "TEC"+
	    ((getTEC()==1)?"-":"+")+
	    " Wheel "+getWheel()+" "+
	    ((getFrontBack()==1)?"back":"front")+" petal "
	    +getPetal()+
	    " Ring "+getRing()+
	    " module "+getModNumber()+
	    ((getStereo()==1)?" Stereo":(getStereo()==0?" Glued":" Mono"));
    }

    public static void main(String args[]){
	try{
	    //System.out.println(args[0]);
	    //TECDetIdConverter d = new TECDetIdConverter(args[0]);
	    TECDetIdConverter d = new TECDetIdConverter(Integer.parseInt(args[0]));
	    System.out.println("Det ID : "+d.compact());

	    System.out.println("Module : "+d.getDetector()+
			       " - "+d.getSubDetector()+
			       " - "+d.getTEC()+
			       " - "+d.getWheel()+
			       " - "+d.getFrontBack()+
			       " - "+d.getPetal()+
			       " - "+d.getRing()+
			       " - "+d.getModNumber()+
			       " - "+d.getStereo()
			       );
	    System.out.println(d);
	}
	catch(Exception e){
	    System.out.println(e.getMessage());
	}
    }

}
