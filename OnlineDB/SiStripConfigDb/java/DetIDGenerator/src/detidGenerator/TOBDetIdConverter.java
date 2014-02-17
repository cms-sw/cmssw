package detidGenerator;

/**
 * <p>Used to convert the det id to a 32 bits word</p>
 * @author G. Baulieu
 * @version 1.0
**/

/*
  $Date: 2007/03/22 20:46:49 $
  
  $Log: TOBDetIdConverter.java,v $
  Revision 1.4  2007/03/22 20:46:49  gbaulieu
  New numbering of Det ID

  Revision 1.3  2006/08/31 15:24:29  gbaulieu
  The TOBCS are directly in the TOB
  Correction on the Stereo flag

  Revision 1.2  2006/08/30 15:21:12  gbaulieu
  Add the TOB analyzer

  Revision 1.1  2006/06/28 11:42:24  gbaulieu
  First import of the sources

  Revision 1.1  2006/02/08 15:03:00  baulieu
  Add the convertion to 32 bits for the TOB


*/

public class TOBDetIdConverter extends DetIdConverter{

    private int layer;
    private int frontBack;
    private int rod;
    private int moduleNumber;
    private int stereo;

    /// two bits would be enough, but  we could use the number "0" as a wildcard
    private final short layerStartBit =     14;
    private final short rod_fw_bwStartBit = 12;
    private final short rodStartBit =       5;
    private final short detStartBit =    2;
    private final short sterStartBit =      0;
    /// two bits would be enough, but  we could use the number "0" as a wildcard

    private final short layerMask =       0x7;
    private final short rod_fw_bwMask =   0x3;
    private final short rodMask =         0x7F;
    private final short detMask =      0x7;
    private final short sterMask =        0x3;


    public TOBDetIdConverter(int l, int fb, int r, int mn, int s){
	super(1, 5);
	layer = l;
	frontBack = fb;
	rod = r;
	moduleNumber = mn;
	stereo = s;
    }

    public TOBDetIdConverter(String detID) throws Exception{
	super(1,5);
	try{
	    String[] val = detID.split("\\.");
	    if(val.length!=7)
		throw new Exception("The detID has an invalid format");
	    else{
		layer = Integer.parseInt(val[2]);
		frontBack = Integer.parseInt(val[3]);
		rod = Integer.parseInt(val[4]);
		moduleNumber = Integer.parseInt(val[5]);
		stereo = Integer.parseInt(val[6]);
	    }
	}
	catch(NumberFormatException e){
	    throw new Exception("TOBDetIdConverter : \n"+e.getMessage());
	}
    }

     public TOBDetIdConverter(int detID) throws Exception{
	super(detID);
	layer = getLayer();
	frontBack = getFrontBack();
	rod = getRod();
	moduleNumber = getModNumber();
	stereo = getStereo();
    }

    public int compact(){
	super.compact();
	id |= (layer&layerMask)<<layerStartBit |
	    (frontBack&rod_fw_bwMask)<<rod_fw_bwStartBit |
	    (rod&rodMask)<<rodStartBit |
	    (moduleNumber&detMask)<<detStartBit |
	    (stereo&sterMask)<<sterStartBit;
	return id;
    }

    public int getLayer(){
	return (id>>layerStartBit)&layerMask;
    }

    public int getFrontBack(){
	return (id>>rod_fw_bwStartBit)&rod_fw_bwMask;
    }

    public int getRod(){
	return (id>>rodStartBit)&rodMask;
    }

    public int getModNumber(){
	return (id>>detStartBit)&detMask;
    }

    public int getStereo(){
	return (id>>sterStartBit)&sterMask;
    }

    public String toString(){
	return "TOB"+
	    ((getFrontBack()==2)?"+":"-")+
	    " Layer "+getLayer()+" "+
	    " Rod "+getRod()+
	    " module "+getModNumber()+
	    ((getStereo()==1)?" Stereo":(getStereo()==0?" Mono":" Glued"));
    }

    public static void main(String args[]){
	try{
	    //System.out.println(args[0]);
	    //TECDetIdConverter d = new TECDetIdConverter(args[0]);
	    TOBDetIdConverter d = new TOBDetIdConverter(Integer.parseInt(args[0]));
	    System.out.println("Det ID : "+d.compact());
	    
	    System.out.println("Module : "+d.getDetector()+
			       " - "+d.getSubDetector()+
			       " - "+d.getLayer()+
			       " - "+d.getFrontBack()+
			       " - "+d.getRod()+
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
