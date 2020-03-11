ROOTINC=-I `root-config --incdir`
ROOTLIBS=`root-config --libs`
#CFLAGS=-std=c++11
CFLAGS=-std=c++1y


#echo "ROOTINC=${ROOTINC}"
#echo "ROOTLIBS=${ROOTLIBS}"

DEPS= LogEleMapdb.h

default: RemoteMonitoringGLOBAL.cc.exe RemoteMonitoringMAP_Global.cc.exe

RemoteMonitoringGLOBAL.cc.exe: RemoteMonitoringGLOBAL.cc $(DEPS)
	c++ $< ${ROOTINC} ${CFLAGS} -o $@ ${ROOTLIBS}

RemoteMonitoringMAP_Global.cc.exe: RemoteMonitoringMAP_Global.cc $(DEPS)
	c++ $< ${ROOTINC} ${CFLAGS} -o $@ ${ROOTLIBS}

clean:
	rm -f *.cc.exe
