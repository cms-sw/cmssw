'''
Created on Nov 16, 2010

@author: MantYdze
'''
import string
import sys
import random
from datetime import datetime

CREATORS = []
REVIEWERS = []
RESPONSIBLES = []
TWIKI_USERS = []    #all emails
TWIKI_USERS_TO_REMIND = []    #only those who have generated reports
PAGES_ADDED = []
ENCODED_URLS = []
ENCODED_USERS = []
TOPICS_TO_SKIP = ["meeting", "minute", "leftbar", "daily"] # patterns to skip   

def addToList(user_email, fields, list):
    line = ",".join(fields)
    if validate_topic(line):
        added = False
        for line_list in list:
            if line_list[0] == user_email:
                line_list.append(line)
                added = True
                break
        if (not added):
            subList = [user_email, line]
            list.append(subList)

def addReviewer(fields):
    reviewer = validate_email(fields[2])
    if (reviewer != None) and (fields[6].strip() == ""):
        addToList(reviewer, fields, REVIEWERS)
    
def addCreator(fields):
    creator = validate_email(fields[5])
    if (creator != None) and (fields[6].strip() == ""):
        addToList(creator, fields, CREATORS)

def addResponsible(fields):
    responsible = validate_email(fields[6])
    if (responsible != None):
        addToList(responsible, fields, RESPONSIBLES)

def validate_topic(topic):
    ''' Returns True if topic is valid, otherwise returns False '''
    for topic_to_skip in TOPICS_TO_SKIP:
        if topic.lower().find(topic_to_skip.lower()) != -1:
            return False
        
    return True

def validate_email(user_email):
    ''' Returns valid user email, otherwise returns None'''
    user_email = user_email.strip().lower()
    if (len(user_email) > 5) and (user_email.find("@") != -1):
        if (not user_email in TWIKI_USERS):
            TWIKI_USERS.append(user_email)
        return user_email
    else:
        return None;

def processFiles():
    cms = open(cmsFilename, "r")
    cmsLines = cms.read().split("\n")
    cms.close()
    
    cmsPublic = open(cmsPublicFilename, "r")
    cmsPublicLines = cmsPublic.read().split("\n")
    cmsPublic.close()
    
    for cmsLine in cmsLines:
        if len(cmsLine) > 20:
            
            cmsFields = cmsLine.split(",")
            if (cmsFields[0] == "1970011"):
                print "CMS "+cmsLine
                # CMSPublic page
                for cmsPublicLine in cmsPublicLines:
                    cmsPublicFields = cmsPublicLine.split(",")
                    # Searching for same title
                    if (cmsPublicFields[1] == cmsFields[1]):
                        print "CMSPublic "+cmsPublicLine
                        cmsPublicFields[1] = "CMSPublic/" + cmsPublicFields[1]
                        cmsFields = cmsPublicFields 
                        break
            else:
                cmsFields[1] = "CMS/" + cmsFields[1]
            
            addCreator(cmsFields)
            addReviewer(cmsFields)
            addResponsible(cmsFields)
            
    TWIKI_USERS.sort()
    
def isOldDate(dateRAW):
    ''' Returns True if date is older that 6 months, otherwise returns False '''
    d = datetime.strptime(dateRAW, "%Y%m%d")
    delta = datetime.now() - d
    if delta.days > PERIOD_DAYS:
        return True
    else:
        return False
    
def formatHtmlFromList(user_email, list):
    html = ""
    
    i = 0;
    for subList in list:
        if subList[0] == user_email:
            for line in subList[1:]:
                # Preventing from duplicate topics in different list
                if (not line in PAGES_ADDED):
                    PAGES_ADDED.append(line)
                    
                    i = i + 1;
                    if (i % 2 == 1):
                        row_class = "odd"
                    else:
                        row_class = "even"
                        
                    row = "<tr align=\"center\" class=\""+row_class+"\">"
                    
                    line_elements = line.split(",")
                    dateRAW = line_elements[0]  # Sorting date
                    title = line_elements[1]
                    titleWithoutDir = title.split("/")[1]
                    reviewer = line_elements[2] # Last Editor
                    dateHR = line_elements[4]   # Date Human Readable
                    creator = line_elements[5]
                    responsible = "" # Default value in case better is not found 
                    
                    for responsible_tmp in line_elements[6:]:
                        responsible_tmp = responsible_tmp.strip().lower()
                        if (len(responsible_tmp) > 5):
                            if responsible_tmp.find("@") != -1:
                                responsible_tmp = "<a href=\"mailto:"+responsible_tmp+"\">"+responsible_tmp+"</a>"
                            responsible += responsible_tmp+"<br/>"
                    
                    date_warning = ""
                    if isOldDate(dateRAW):
                        date_warning = " style=\"color: red\""
                    
                    creator_warning = ""
                    creatorHTML = creator
                    if (creator.find("@") == -1):
                        creator_warning = " style=\"color: red\""
                    else:
                        creatorHTML = "<a target=\"_blank\" href=\"mailto:"+creator+"\">"+creator+"</a>"
                    
                    reviewerHTML = "<a target=\"_blank\" href=\"mailto:"+reviewer+"\">"+reviewer+"</a>"
                    titleHTML = "<a target=\"_blank\" href=\""+BASE+title+"\">"+titleWithoutDir+"</a>"
                    
                    row += "<td>"+responsible+"</td>"
                    row += "<td>"+titleHTML+"</td>"
                    row += "<td"+date_warning+">"+dateHR+"</td>"
                    row += "<td>"+reviewerHTML+"</td>"
                    row += "<td"+creator_warning+">"+creatorHTML+"</td>"
                    row += "<td><input type=\"checkbox\" value=\"0\" name=\""+title+"\"></td>" 
                    
                    row += "</tr>\n"
                    
                    html += row
            break
    return html

def encode_url(user_email):
    cipher = ''.join(random.choice(string.letters) for i in range(20))
    ENCODED_URLS.append(cipher+" "+user_email)
    
#    return cipher
    return user_email

def encode_user(user_email):
    cipher = ''.join(random.choice(string.letters) for i in range(20))
    ENCODED_USERS.append(cipher+" "+user_email)
#    return cipher
    return user_email

def saveTwikiUsers(list, filename):
    output = open(filename, 'w+')
    output.write("\n".join(list))
    output.close()
    
def createHtmlReport(user_email):
    global PAGES_ADDED
    PAGES_ADDED = []

    input = open("report_template.html", "r")
    html = input.read()
    input.close()
        
    hasOldPages = False
    
    responsibleHTML = formatHtmlFromList(user_email, RESPONSIBLES)
    html = html.replace("REPLACE_RESPONSIBLE", responsibleHTML)
    if (responsibleHTML != ""):
        hasOldPages = True
            
    reviewerHTML = formatHtmlFromList(user_email, REVIEWERS)
    html = html.replace("REPLACE_REVIEWER", reviewerHTML)
    if (reviewerHTML != ""):
        hasOldPages = True
        
    creatorHTML = formatHtmlFromList(user_email, CREATORS)
    html = html.replace("REPLACE_CREATOR", creatorHTML)
    if (creatorHTML != ""):
        hasOldPages = True
        
    if (hasOldPages):
        html = html.replace("REPLACE_SENDER", encode_user(user_email))
        html = html.replace("REPLACE_USERNAME", user_email.split("@")[0])
        html = html.replace("REPLACE_DATE_PROVIDED", DATE_PROVIDED)
        
        output = open("reports/"+encode_url(user_email)+"_report.html",'w')
        output.write(html)
        output.close()
        
        TWIKI_USERS_TO_REMIND.append(user_email)

global cmsFilename
cmsFilename = "CMS.txt"

global cmsPublicFilename
cmsPublicFilename = "CMSPublic.txt"

global DATE_PROVIDED
DATE_PROVIDED = "2013-04-22"

global PERIOD_DAYS
PERIOD_DAYS = 180

global BASE
BASE = "https://twiki.cern.ch/twiki/bin/viewauth/"

processFiles()

for user_email in TWIKI_USERS:
    print user_email+"\n"
    createHtmlReport(user_email)
    
saveTwikiUsers(TWIKI_USERS_TO_REMIND, "twiki_users201304.txt")
        
