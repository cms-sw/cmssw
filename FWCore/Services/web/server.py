from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import mimetypes

class Serv(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        
        try:
            file_path = self.path[1:]

            with open(file_path, 'rb') as file:
                file_to_open = file.read()

            mime_type, _ = mimetypes.guess_type(file_path)

            self.send_response(200)
            if mime_type:
                self.send_header("Content-type", mime_type)
            self.end_headers()

            self.wfile.write(file_to_open)
        
        except FileNotFoundError:
            self.send_response(404)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes("File not found", 'utf-8'))
        
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes(f"Internal server error: {str(e)}", 'utf-8'))

httpd = HTTPServer(('localhost', 65432), Serv)
print("Server started at http://localhost:65432")
httpd.serve_forever()
