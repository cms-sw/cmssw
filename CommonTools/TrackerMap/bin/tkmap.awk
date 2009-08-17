BEGIN{
    n=0;
    endedParsing=0;
    selfArrange=0;
    min=99999999999;
    max=-1*min;
    selfArrange=2;
}
function getcolor(val){
    red=0;green=0;blue=0;
    delta=(max-min);
    x =(val-min);
    if(val<min){red=0;green=0;blue=255;}
    if(val>max){red=255;green=0;blue=0;}
    if(val>=min&&val<=max){ 
	red = (int) ( x<(delta/2) ? 0 : ( x > ((3./4.)*delta) ?  255 : 255/(delta/4) * (x-(2./4.)*delta)  ) );
	green= (int) ( x<delta/4 ? (x*255/(delta/4)) : ( x > ((3./4.)*delta) ?  255-255/(delta/4) *(x-(3./4.)*delta) : 255 ) );
	blue = (int) ( x<delta/4 ? 255 : ( x > ((1./2.)*delta) ?  0 : 255-255/(delta/4) * (x-(1./4.)*delta) ) );
    }
    color=sprintf("%d,%d,%d",red,green,blue);
    return color; 
}
function parseSource(){
    value[$1]=$2;
    print min " " max;
    if(selfArrange==1){   
	if($2<min)
	    min=$2;
	if($2>max)
	    max=$2;
    }
}
function fillTkMap(){
    if(match($2,"detid")){
	det=substr($2,8,9);
	if(det in value){
	    tmp=$0
	    stringVal=sprintf("value=\"%f\"",value[det]);
	    sub("255,255,255", getcolor(value[det]), tmp);
	    sub("value=\"0\"", stringVal, tmp);
	    print tmp
	} else 
	    print $0
    } else
	print $0
}
{
    if(selfArrange==2){
	selfArrange=1;
	if(vmin!=vmax){
	    min=vmin;
	    max=vmax;
	    selfArrange=0;
	}
    }
    if(!match(FILENAME,"tkmap_white.xml")) parseSource();
    
    if(match(FILENAME,"tkmap_white.xml")) fillTkMap();
}