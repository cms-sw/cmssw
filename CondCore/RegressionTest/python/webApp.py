import os
import cherrypy
import web_results_display
import service

sessionsDirectory = os.path.join(service.settings['rootDirectory'], 'sessions')

class Server:
		
	def ShowTable(self, curs, label):
		res1 = web_results_display.GetRunResults(curs[0])
		# only 
		for rows in res1:
			res2, stCount = web_results_display.GetResultsList(rows[0])
			break
		#print stCount
		Code ="""
		        <table id="header" bgcolor="#D8D8D8">
					<tr>
					<td colspan = "2">Test Sequence:
					<b>"""+str(curs[2])+"""</b>&nbsp&nbsp&nbsp&nbsp
					RunID:
					"""+str(curs[0])+"""&nbsp&nbsp&nbsp&nbsp
					Time:
					"""+str(curs[1])+"""
					</td>
					<tr>
					<td>
					Candidate:
					<b>"""+str(curs[3])+"""</b>&nbsp&nbsp&nbsp&nbsp
					Architecture:
					<b>"""+str(curs[4])+"""</b>
					</td>
					<td class="links"><a href="showLogs?runID="""+str(curs[0])+"""&label="""+label+"""">&gt&gt Logfile</a></td>
					</tr>
			</table>
			<table id="status">
			"""
		Code +="""
					<tr>
						<!--<th rowspan="2" colspan = "2">Reference releases</th>-->
						<th>Reference release</th>
						<th>Reference architecture</th>
					
		"""
		for i in range (0, stCount):
			Code += """<th>"""+str(res2[i][2])+"""</th>
			"""
		Code +="""</tr>"""
		for rows in res1:
		        res2, stCount = web_results_display.GetResultsList(rows[0])
			Code += """<tr>
					<td align="left" >"""+str(rows[1])+"""</td>
					<td>"""+str(rows[2])+"""</td>
			"""
			for i in range (0, stCount):
				if(res2[i][1] == 0):
					Code +="""<td align="center" bgcolor ="#A7C942"><b>OK</b></td>
					"""
				else:
					Code +="""<td align="center" bgcolor ="#FF0000"><b>Failure</b></td>
					"""
			Code += """</tr></tr>
			"""
		Code += """</table><hr>
		"""
		Code += """
		</body>
		</html>
		"""
		return Code
	def index(self, **args):
		print "Arguments :"
		print args
		selectRel = False
		if 'reset' in args.keys():
			cherrypy.session.clear()
		it = 0
		for row in web_results_display.GetLabels():
			if not 'label' in cherrypy.session.keys():
				cherrypy.session['label'] = row[0]
			break
		if 'submit' in args.keys():
			cherrypy.session['submit'] = args['submit']
			cherrypy.session['label'] = args['label']
			if 'release' in args.keys() and args['release'] != '':
				cherrypy.session['release'] = args['release']
			if 'release' in cherrypy.session.keys() and args['release'] != cherrypy.session['release']:
				cherrypy.session['release'] = args['release']
			if 'arch' in args.keys() and args['arch'] != '':
				cherrypy.session['arch'] = args['arch']
			if 'arch' in cherrypy.session.keys() and args['arch'] != cherrypy.session['arch']:
				cherrypy.session['arch'] = args['arch']
			if 'count' in args.keys() and args['count'].isdigit():
				cherrypy.session['count'] = args['count']
			print "Session :"	
			print cherrypy.session
		if 'arch' in cherrypy.session.keys() and cherrypy.session['arch'] != ''  or 'release' in cherrypy.session.keys() and cherrypy.session['release'] != '':
			selectRel = True
		if 'count' in cherrypy.session.keys() and cherrypy.session['count'].isdigit():
				stCount = int(cherrypy.session['count'])
		else:
			stCount = 2;
		
		htmlCode = """
		<html>
			<META HTTP-EQUIV="REFRESH" CONTENT="60">
			<head>
				<style type="text/css">
					button,
					input,
					select,
					textarea {
					  font-size: 100%;
					  margin: 0;
					  vertical-align: baseline;
					  *vertical-align: middle;
					}
					button, input {
					  line-height: normal;
					  *overflow: visible;
					}
					button::-moz-focus-inner, input::-moz-focus-inner {
					  border: 0;
					  padding: 0;
					}
					select {
					}
					button,
					input[type="submit"] {
					  cursor: pointer;
					  -webkit-appearance: button;
					}
					input[type="search"] {
					  -webkit-appearance: textfield;
					  -webkit-box-sizing: content-box;
					  -moz-box-sizing: content-box;
					  box-sizing: content-box;
					}
					a {
					  color: #bfbfbf;
					  text-shadow: 0 -1px 0 rgba(0, 0, 0, 0.25);
					  text-decoration:none;
					}
					a:hover, .active a {
					  color: #ffffff;
					  text-decoration: none;
					}
					.links {
					  background-color: #222;
					  background-color: #222222;
					  background-repeat: repeat-x;
					  background-image: -khtml-gradient(linear, left top, left bottom, from(#333333), to(#222222));
					  background-image: -moz-linear-gradient(top, #333333, #222222);
					  background-image: -ms-linear-gradient(top, #333333, #222222);
					  background-image: -webkit-gradient(linear, left top, left bottom, color-stop(0%, #333333), color-stop(100%, #222222));
					  background-image: -webkit-linear-gradient(top, #333333, #222222);
					  background-image: -o-linear-gradient(top, #333333, #222222);
					  background-image: linear-gradient(top, #333333, #222222);
					  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#333333', endColorstr='#222222', GradientType=0);
					  -webkit-box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25), inset 0 -1px 0 rgba(0, 0, 0, 0.1);
					  -moz-box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25), inset 0 -1px 0 rgba(0, 0, 0, 0.1);
					  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25), inset 0 -1px 0 rgba(0, 0, 0, 0.1);
					}
					#status, #header
					{
						font-family:"Trebuchet MS", Arial, Helvetica, sans-serif;
						width:100%;
						border-collapse:collapse;
					}
					#header td 
					{
						font-size:1em;
						border:1px solid #67645B;
						padding:3px 7px 2px 7px;
						text-align:left;
					}
					#status td
					{
						font-size:1em;
						border:1px solid #67645B;
						padding:3px 7px 2px 7px;
					}
					#status th 
					{
					        border:1px solid #67645B;
						font-size:1.1em;
						text-align:center;
						padding-top:5px;
						padding-bottom:4px;
						background-color:#ECAE12;
						color:#ffffff;
					}
					#controls
					{
						font-family:"Trebuchet MS", Arial, Helvetica, sans-serif;
						width:100%;
						border-collapse:collapse;
					}
					#controls td 
					{
						font-size:1.1em;
						border:none;
						padding-top:5px;
						padding-bottom:4px;
						text-align:left;
						background-color:#393333;
						color:#ffffff;
						
					}
					.topbar {
					  height: 40px;
					  position: fixed;
					  top: 0;
					  left: 0;
					  right: 0;
					  z-index: 10000;
					  overflow: visible;
					  color: #ffffff;
					  width: 100%;
					}
					.topbar form {
					  float: left;
					  margin: 5px 0 0 0;
					  position: relative;
					  filter: alpha(opacity=100);
					  -khtml-opacity: 1;
					  -moz-opacity: 1;
					  opacity: 1;
					}
					.topbar input {
					  background-color: #444;
					  background-color: rgba(255, 255, 255, 0.3);
					  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
					  font-size: 80%;
					  font-weight: 13px;
					  line-height: 1;
					  padding: 4px 9px;
					  color: #fff;
					  color: rgba(255, 255, 255, 0.75);
					  border: 1px solid #111;
					  -webkit-border-radius: 4px;
					  -moz-border-radius: 4px;
					  border-radius: 4px;
					  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1), 0 1px 0px rgba(255, 255, 255, 0.25);
					  -moz-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1), 0 1px 0px rgba(255, 255, 255, 0.25);
					  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1), 0 1px 0px rgba(255, 255, 255, 0.25);
					  -webkit-transition: none;
					  -moz-transition: none;
					  transition: none;
					}
					.topbar input:-moz-placeholder {
					  color: #e6e6e6;
					}
					.topbar input::-webkit-input-placeholder {
					  color: #e6e6e6;
					}
					.topbar input:hover {
					  background-color: #bfbfbf;
					  background-color: rgba(255, 255, 255, 0.5);
					  color: #fff;
					}
					.topbar input:focus, .topbar input.focused {
					  outline: none;
					  background-color: #fff;
					  color: #404040;
					  text-shadow: 0 1px 0 #fff;
					  border: 0;
					  padding: 5px 10px;
					  -webkit-box-shadow: 0 0 3px rgba(0, 0, 0, 0.15);
					  -moz-box-shadow: 0 0 3px rgba(0, 0, 0, 0.15);
					  box-shadow: 0 0 3px rgba(0, 0, 0, 0.15);
					}
					.topbar-inner, .topbar .fill {
					  background-color: #222;
					  background-color: #222222;
					  background-repeat: repeat-x;
					  background-image: -khtml-gradient(linear, left top, left bottom, from(#333333), to(#222222));
					  background-image: -moz-linear-gradient(top, #333333, #222222);
					  background-image: -ms-linear-gradient(top, #333333, #222222);
					  background-image: -webkit-gradient(linear, left top, left bottom, color-stop(0%, #333333), color-stop(100%, #222222));
					  background-image: -webkit-linear-gradient(top, #333333, #222222);
					  background-image: -o-linear-gradient(top, #333333, #222222);
					  background-image: linear-gradient(top, #333333, #222222);
					  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#333333', endColorstr='#222222', GradientType=0);
					  -webkit-box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25), inset 0 -1px 0 rgba(0, 0, 0, 0.1);
					  -moz-box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25), inset 0 -1px 0 rgba(0, 0, 0, 0.1);
					  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25), inset 0 -1px 0 rgba(0, 0, 0, 0.1);
					}
					.buts {
					float: right;
					}
				</style>
			</head>
			<body>
				<form method="post" action="index" >
						<div class="topbar">
							<div class="fill">
								<div class="container">
									<select name = "label">"""
		curs = web_results_display.GetLabels()
		for row in curs:
			if str(row[0]) == cherrypy.session['label']:
				htmlCode += """<option value = """+str(row[0])+""" selected = "selected">"""+str(row[0])+"""</option>
				"""
			else:
				htmlCode += """<option value = """+str(row[0])+""">"""+str(row[0])+"""</option>
				"""
		htmlCode+="""</select>
			"""
		if 'release' in cherrypy.session.keys():
			htmlCode += """Candidate Release:<input type="text" name="release" size="30" maxlength="50" value=\""""+str(cherrypy.session['release'])+""""\"/>
			"""
		else:
			htmlCode += """Candidate Release:<input type="text" name="release" size="30" maxlength="50"/>
			"""
		if 'arch' in cherrypy.session.keys():
			htmlCode += """Candidate Architecture:<input type="text" name="arch" size="15" maxlength="30" value=\""""+cherrypy.session['arch']+""""\"/>
			"""
		else:
			htmlCode += """Candidate Architecture:<input type="text" name="arch" size="15" maxlength="30" />
			"""
		if 'count' in cherrypy.session.keys():
			htmlCode += """Number of results:<input type="text" name="count" size="1" maxlength="4" value=\""""+cherrypy.session['count']+""""\"/>
			"""
		else:
			htmlCode += """Number of results:<input type="text" name="count" size="4" maxlength="4" />
			"""
		htmlCode +="""<div class = "buts"><input type="submit" name="submit" value="Refresh"/>
							<input type="submit" name="reset" value="Reset"/>
						</div></div></div></div>
						<br>
				</form>
			"""
		if(selectRel == True):
			if 'arch' in cherrypy.session.keys() and 'release' in cherrypy.session.keys():
				DBdata = web_results_display.GetReleasesHeaders(cherrypy.session['label'], cherrypy.session['release'], cherrypy.session['arch'])
			elif 'release' in cherrypy.session.keys():
				DBdata = web_results_display.GetReleasesHeaders(cherrypy.session['label'], cherrypy.session['release'])
			elif 'arch' in cherrypy.session.keys():
				DBdata = web_results_display.GetReleasesHeaders(cherrypy.session['label'], "", cherrypy.session['arch'])
			counter =0
			for data in DBdata:
				counter += 1
				if stCount != 0:
					htmlCode += self.ShowTable( data, cherrypy.session['label'])
					stCount = stCount -1
			if (counter == 0):
				htmlCode += """<h3> No entries found </h3>"""
		else:
			DBdata = web_results_display.GetResultHeaders(cherrypy.session['label'])
			for data in DBdata:
			        print 'run=',data[0]
				if stCount != 0:
					htmlCode += self.ShowTable(data, cherrypy.session['label'])
					stCount = stCount -1
		return htmlCode

	index.exposed = True
	def showLogs(self, **args):
		runID = int(args['runID'])
		label = args['label']

		logInfo = "<html><head></head><body><pre>\n"
		logInfo += web_results_display.GetReadLogStatus(label, runID)
		logInfo += '\n</pre></body></html>'
		return logInfo
				
	showLogs.exposed = True

def main():
	service.start(Server())


if __name__ == '__main__':
	main()


