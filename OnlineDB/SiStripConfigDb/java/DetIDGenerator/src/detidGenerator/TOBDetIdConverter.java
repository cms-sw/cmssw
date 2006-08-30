package detidGenerator;

/**
 * <p>Used to convert the det id to a 32 bits word</p>
 * @author G. Baulieu
 * @version 1.0
**/

/*
  $Date: 2006/06/28 11:42:24 $
  
  $Log: TOBDetIdConverter.java,v $
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

    private final short layerStartBit =     16;
    private final short rod_fw_bwStartBit = 15;
    private final short rodStartBit =        8;
    private final short detStartBit =        2;
    private final short sterStartBit =       0;

    private final short layerMask =       0xF;
    private final short rod_fw_bwMask =   0x1;
    private final short rodMask =         0x7F;
    private final short detMask =         0x3F;
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
	    " Layer "+getLayer()+" "+
	    ((getFrontBack()==1)?"forward":"backward")+" half"+
	    " Rod "+getRod()+
	    " module "+getModNumber()+
	    ((getStereo()==1)?" Stereo":(getStereo()==0?" Glued":" Mono"));
    }

}